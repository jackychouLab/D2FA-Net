import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cruw import CRUW
from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.visualization import visualize_train_img
import random
import warnings
import shutil
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.post_processing import post_process_single_frame
from rodnet.core.post_processing import write_dets_results_single_frame
from tqdm import tqdm
from cruw.eval.rod.rod_eval_utils import accumulate, summarize
from cruw.eval import evaluate_rodnet_seq
import matplotlib
import torch.nn.functional as F
import copy
matplotlib.use("Agg")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def set_seed(seed, use_benchmark=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if use_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Seed is setting to {seed}")


class bcel_soft(nn.Module):
    def __init__(self, soft_beta=1.0, reduction='mean'):
        super(bcel_soft, self).__init__()
        self.beta = soft_beta
        self.reduction = reduction

    def forward(self, predict, target):
        bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        pos_loss = target.eq(1) * bce_loss
        neg_loss = (1 - target).pow(self.beta) * bce_loss
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError('Unknown reduction {}'.format(self.reduction))


class smooth_l1l(nn.Module):
    def __init__(self, reduction='mean'):
        super(smooth_l1l, self).__init__()
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        return F.smooth_l1_loss(predict, target, reduction=self.reduction)


def e_data_augment(input_tensor, label_tensor):
    dice = np.random.rand()
    if dice <= 0.5:
        input_tensor = torch.flip(input_tensor, dims=[2])
        label_tensor = torch.flip(label_tensor, dims=[2])
    if dice <= 0.25:
        input_tensor = torch.flip(input_tensor, dims=[-1])
        label_tensor = torch.flip(label_tensor, dims=[-1])
    elif 0.25 < dice <= 0.5:
        noise = torch.randn(input_tensor.shape) * np.std(input_tensor.detach().numpy()) * 0.1 + np.mean(input_tensor.detach().numpy())
        input_tensor = input_tensor + noise
    elif 0.5 < dice <= 0.75:
        noise = torch.randn(input_tensor.shape) * np.std(input_tensor.detach().numpy()) * 0.1 + np.mean(input_tensor.detach().numpy())
        input_tensor = input_tensor + noise
        input_tensor = torch.flip(input_tensor, dims=[-1])
        label_tensor = torch.flip(label_tensor, dims=[-1])
    return input_tensor, label_tensor


class EMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, ema_v in ema_state.items():
            model_v = model_state[k].detach()

            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


class Trainer:
    def __init__(self, config_dict, args):
        # 初始化
        self.config_dict = config_dict
        self.args = args
        self.code_dir = self.args.code_dir
        self.dataset = CRUW(data_root=self.config_dict['dataset_cfg']['base_root'], sensor_config_name=os.path.join(self.code_dir, 'configs/CRUW/dataset_configs/sensor_config'), object_config_name=os.path.join(self.code_dir, 'configs/CRUW/dataset_configs/object_config'))
        self.radar_configs = self.dataset.sensor_cfg.radar_cfg
        self.range_grid = self.dataset.range_grid
        self.angle_grid = self.dataset.angle_grid
        self.model_cfg = self. config_dict['model_cfg']
        self.batch_size = self.config_dict['train_cfg']['batch_size']
        self.train_configs = self.config_dict['train_cfg']
        self.test_configs = self.config_dict['test_cfg']
        self.dataset_configs = self.config_dict['dataset_cfg']
        self.optim_configs = self.config_dict['optim_cfg']
        self.schedule_configs = self.config_dict['schedule_cfg']
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.train_model_path = self.args.log_dir
        self.model_dir, self.model_name = create_dir_for_new_model(self.model_cfg['name'], self.train_model_path)
        self.train_viz_path = os.path.join(self.model_dir, 'train_viz')
        if not os.path.exists(self.train_viz_path):
            os.makedirs(self.train_viz_path)
        self.writer = SummaryWriter(self.model_dir)
        self.save_config_dict = {
            'args': vars(self.args),
            'config_dict': self.config_dict,
        }
        self.config_json_name = os.path.join(self.model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
        with open(self.config_json_name, 'w') as fp:
            json.dumps(self.save_config_dict, cls=NpEncoder)
        self.train_log_name = os.path.join(self.model_dir, "train.log")
        with open(self.train_log_name, 'w'):
            pass
        self.crdata_train = CRDataset(data_dir=args.data_dir, dataset=self.dataset, config_dict=config_dict, split='train', noise_channel=args.use_noise_channel, data_aug=self.model_cfg['data_aug'], data_norm=self.model_cfg['data_norm'], is_random_chirp=True)
        self.seq_names = self.crdata_train.seq_names
        self.index_mapping = self.crdata_train.index_mapping
        self.train_dataloader = DataLoader(self.crdata_train, self.batch_size, shuffle=True, num_workers=self.config_dict['train_cfg']['num_workers'], persistent_workers=True, pin_memory=True, prefetch_factor=2)
        self.eval_seq_names = self.dataset_configs['test']['seqs']
        self.eval_dataloader_list = {}
        for subset_idx in self.eval_seq_names:
            subset = f'{subset_idx}'
            crdata_test = CRDataset(data_dir=self.args.data_dir, dataset=self.dataset, config_dict=self.config_dict, split='test', noise_channel=self.args.use_noise_channel, subset=subset, is_random_chirp=False)
            eval_dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=self.config_dict['train_cfg']['num_workers'], persistent_workers=True, collate_fn=cr_collate)
            self.eval_dataloader_list[subset] = eval_dataloader

        # 加载模型
        if self.model_cfg['type'] == 'CDC':
            from rodnet.models import RODNetCDC as MODEL
        elif self.model_cfg['type'] == 'HG':
            from rodnet.models import RODNetHG as MODEL
        elif self.model_cfg['type'] == 'HGwI':
            from rodnet.models import RODNetHGwI as MODEL
        elif self.model_cfg['type'] == 'CDCv2':
            from rodnet.models import RODNetCDCDCN as MODEL
        elif self.model_cfg['type'] == 'HGv2':
            from rodnet.models import RODNetHGDCN as MODEL
        elif self.model_cfg['type'] == 'HGwIv2':
            from rodnet.models import RODNetHGwIDCN as MODEL
        elif self.model_cfg['type'] == 'T-RODNet':
            from rodnet.models import T_RODNet as MODEL
        elif self.model_cfg['type'] == 'E-RODNet':
            from rodnet.models import E_RODNet as MODEL
        elif self.model_cfg['type'] == 'STCT':
            from rodnet.models import STCTNet as MODEL
        elif self.model_cfg['type'] == 'D2FA-Net':
            from rodnet.models import D2FANet as MODEL
        elif self.model_cfg['type'] == 'DCSN':
            from rodnet.models import RODNet_DCSN as MODEL
        elif self.model_cfg['type'] == 'SS-RODNet':
            from rodnet.models import SS_RODNet as MODEL
        elif self.model_cfg['type'] == 'Mask-RadarNet':
            from rodnet.models import MaskRadar as MODEL
        else:
            raise NotImplementedError

        self.n_class = self.dataset.object_cfg.n_class
        self.confmap_shape = (self.n_class, 128, 128)
        self.n_epoch = config_dict['schedule_cfg']['n_epoch']
        self.lr = config_dict['optim_cfg']['lr']

        if 'stacked_num' in self.model_cfg:
            self.stacked_num = self.model_cfg['stacked_num']
        else:
            self.stacked_num = None

        if args.use_noise_channel:
            self.n_class_train = self.n_class + 1
        else:
            self.n_class_train = self.n_class

        print("Building model ... (%s)" % self.model_cfg)

        if self.model_cfg['type'] == 'CDCv2':
            self.in_chirps = len(self.radar_configs['chirp_ids'])
            self.model = MODEL(in_channels=self.in_chirps, n_class=self.n_class_train, mnet_cfg=config_dict['model_cfg']['mnet_cfg'], dcn=config_dict['model_cfg']['dcn']).cuda()
        elif self.model_cfg['type'] == 'HGv2':
            self.in_chirps = len(self.radar_configs['chirp_ids'])
            self.model = MODEL(in_channels=self.in_chirps, n_class=self.n_class_train, stacked_num=self.stacked_num, mnet_cfg=config_dict['model_cfg']['mnet_cfg'], dcn=config_dict['model_cfg']['dcn']).cuda()
        elif self.model_cfg['type'] == 'HGwIv2':
            self.in_chirps = len(self.radar_configs['chirp_ids'])
            self.model = MODEL(in_channels=self.in_chirps, n_class=self.n_class_train, stacked_num=self.stacked_num, mnet_cfg=config_dict['model_cfg']['mnet_cfg'], dcn=config_dict['model_cfg']['dcn']).cuda()
        elif self.model_cfg['type'] == 'E-RODNet':
            self.model = MODEL(self.config_dict['model_cfg']['mnet_cfg'], self.n_class_train).cuda()
        elif self.model_cfg['type'] == 'T-RODNet':
            self.model = MODEL(num_classes=self.n_class_train, embed_dim=64, win_size=4).cuda()
        elif self.model_cfg['type'] == 'STCT':
            self.model = MODEL(self.config_dict['model_cfg']['mnet_cfg'], self.n_class_train).cuda()
        elif self.model_cfg['type'] == 'DCSN':
            self.model = MODEL(self.n_class_train).cuda()
        elif self.model_cfg['type'] == 'Mask-RadarNet':
            self.model = MODEL(self.n_class_train).cuda()
        elif self.model_cfg['type'] == 'D2FA-Net':
            self.model = MODEL(self.config_dict['model_cfg']['mnet_cfg'], self.n_class_train, self.config_dict['model_cfg']['depths'], self.config_dict['model_cfg']['channels'], self.config_dict['model_cfg']['drop_rate'], mnet_type=self.config_dict['model_cfg']['mnet_type'], tokenMixer=self.config_dict['model_cfg']['tokenMixer'], fpn_channel=self.config_dict['model_cfg']['fpn_channel'], filter_cfg=self.config_dict['model_cfg']['filter_cfg'], head_num=self.config_dict['model_cfg']['head_num']).cuda()
        elif self.model_cfg['type'] == 'SS-RODNet':
            self.model = MODEL(num_classes=self.n_class_train, embed_dim=64, win_size=4, pretraining=False)
            checkpoint = torch.load(os.path.join(self.code_dir, 'configs/CRUW/CE/ss-rodnet-pretrain.pkl'), map_location='cpu')
            model_state_dict = self.model.state_dict()
            matched_checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            missing_keys, unexpected_keys = self.model.load_state_dict(matched_checkpoint, strict=False)
            del checkpoint
            del matched_checkpoint
            del missing_keys, unexpected_keys
            self.model = self.model.cuda()
        else:
            raise TypeError

        if self.model_cfg['loss_type'] == 'bcel':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.model_cfg['loss_type'] == 'soft_bcel':
            self.criterion = bcel_soft(reduction='mean')
        elif self.model_cfg['loss_type'] == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        elif self.model_cfg['loss_type'] == 'smooth_l1l':
            self.criterion = smooth_l1l(reduction='mean')
        elif self.model_cfg['loss_type'] == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise TypeError

        iter_num = len(self.train_dataloader)
        print(f"Each epoch has {iter_num} iterations.")

        if self.schedule_configs['type'] == 'cosine_iter':
            if self.optim_configs['type'] == 'adamw':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optim_configs['lr'], eps=1e-6, weight_decay=1e-3)
            elif self.optim_configs['type'] == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.optim_configs['lr'])
            else:
                raise TypeError
            self.scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[LinearLR(self.optimizer, start_factor=1e-7 / self.optim_configs['lr'], total_iters=self.schedule_configs['warmup_epoch'] * iter_num), CosineAnnealingLR(self.optimizer, int(self.n_epoch * iter_num - self.schedule_configs['warmup_epoch'] * iter_num), eta_min=1e-6)], milestones=[self.schedule_configs['warmup_epoch'] * iter_num])
        elif self.schedule_configs['type'] == 'cosine_epoch':
            if self.optim_configs['type'] == 'adamw':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optim_configs['lr'])
            elif self.optim_configs['type'] == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.optim_configs['lr'])
            else:
                raise TypeError
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epoch)
        elif self.schedule_configs['type'] == 'cosine_epoch_warmup':
            if self.optim_configs['type'] == 'adamw':
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optim_configs['lr'])
            elif self.optim_configs['type'] == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.optim_configs['lr'])
            else:
                raise TypeError
            self.scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[LinearLR(self.optimizer, start_factor=1e-7 / self.optim_configs['lr'], total_iters=self.schedule_configs['warmup_epoch']), CosineAnnealingLR(self.optimizer, int(self.n_epoch - self.schedule_configs['warmup_epoch']))], milestones=[self.schedule_configs['warmup_epoch']])
        elif self.schedule_configs['type'] == 'base-rodnet':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optim_configs['lr'])
            self.scheduler = StepLR(self.optimizer, step_size=self.optim_configs['lr_step'], gamma=0.1)
        else:
            raise TypeError

        if self.train_configs.get('use_ema', False):
            self.ema = EMA(self.model)
        else:
            self.ema = None

        self.iter_count = 0
        self.loss_ave = 0
        self.best_map = 0.
        self.best_mar = 0.
        self.best_ap50 = 0.
        self.best_ap70 = 0.
        self.best_score = 0.
        self.best_score_epoch = -1
        self.all_iters = len(self.train_dataloader) * self.n_epoch
        self.patience_count = 0

    def train(self,):
        for epoch in range(self.n_epoch):
            self.model.train()
            for iter, data_dict in enumerate(self.train_dataloader):
                tic_load = time.perf_counter()
                data = data_dict[0]
                confmap_gt = data_dict[1]
                image_paths = data_dict[2]
                if self.model_cfg['e_data_aug']:
                    data, confmap_gt = e_data_augment(data, confmap_gt)
                visual_data = data
                visual_gt = confmap_gt
                data = data.to(device="cuda", dtype=torch.float32, non_blocking=True)
                confmap_gt = confmap_gt.to(device="cuda", dtype=torch.float32, non_blocking=True)
                tic = time.perf_counter()
                self.optimizer.zero_grad()
                if self.model_cfg['type'] == 'Mask-RadarNet':
                    confmap_preds, exo_preds = self.model(data)
                else:
                    confmap_preds = self.model(data)
                loss_confmap = 0
                if self.stacked_num is not None:
                    for i in range(self.stacked_num):
                        loss_cur = self.criterion(confmap_preds[i], confmap_gt)
                        loss_confmap += loss_cur
                else:
                    if type(confmap_preds) is list:
                        for i in range(len(confmap_preds)):
                            loss_confmap += self.criterion(confmap_preds[i], confmap_gt)
                    else:
                        loss_confmap = self.criterion(confmap_preds, confmap_gt)
                        if self.model_cfg['type'] == 'Mask-RadarNet':
                            loss_confmap += self.criterion(exo_preds, confmap_gt) * 0.4
                loss_confmap.backward()
                tic_back = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 35.0, 2.0, False)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                if 'iter' in self.schedule_configs['type']:
                    self.scheduler.step()
                self.all_iters -= 1
                if iter % self.config_dict['train_cfg']['log_step'] == 0:
                    load_time = tic - tic_load
                    back_time = tic_back - tic
                    wait_time = (load_time + back_time) * self.all_iters
                    wait_h, wait_r = divmod(wait_time, 3600)
                    wait_m, wait_s = divmod(wait_r, 60)
                    print('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.4f | back time: %.4f | lr: %.8f | best AP50 acc: %.4f | best AP70 acc: %.4f | best mAP acc: %.4f | best mAR acc: %.4f | best Score: %.4f | best Score epoch: %2d | wait time: %2d : %2d : %2d' % (epoch + 1, iter + 1, loss_confmap.item(), self.loss_ave, load_time, back_time, self.optimizer.param_groups[0]['lr'], self.best_ap50, self.best_ap70, self.best_map, self.best_mar, self.best_score, self.best_score_epoch, wait_h, wait_m, wait_s))
                    with open(self.train_log_name, 'a+') as f_log:
                        f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.4f | back time: %.4f | lr: %.8f | best AP50 acc: %.4f | best AP70 acc: %.4f | best mAP acc: %.4f | best mAR acc: %.4f | best Score: %.4f | best Score epoch: %2d \n' % (epoch + 1, iter + 1, loss_confmap.item(), self.loss_ave, load_time, back_time, self.optimizer.param_groups[0]['lr'], self.best_ap50, self.best_ap70, self.best_map, self.best_mar, self.best_score, self.best_score_epoch))

                    self.writer.add_scalar('loss/loss_all', loss_confmap.item(), self.iter_count)
                    self.writer.add_scalar('time/time_load', load_time, self.iter_count)
                    self.writer.add_scalar('time/time_back', back_time, self.iter_count)
                    self.writer.add_scalar('param/param_lr', self.scheduler.get_last_lr()[0], self.iter_count)
                    if self.stacked_num is not None:
                        confmap_pred = confmap_preds[self.stacked_num - 1]
                    else:
                        if type(confmap_preds) is list:
                            confmap_pred = confmap_preds[-1]
                        else:
                            confmap_pred = confmap_preds
                    if 'bcel' in self.model_cfg['loss_type'] or 'smooth_l1l' in self.model_cfg['loss_type']:
                        confmap_pred = confmap_pred.sigmoid()
                    confmap_pred = confmap_pred.cpu().detach().numpy()
                    if 'mnet_cfg' in self.model_cfg:
                        chirp_amp_curr = chirp_amp(visual_data[0, :, 0, 0, :, :], self.radar_configs['data_type'])
                    else:
                        chirp_amp_curr = chirp_amp(visual_data[0, :, 0, :, :], self.radar_configs['data_type'])
                    fig_name = os.path.join(self.train_viz_path, '%03d_%010d_%06d.png' % (epoch + 1, self.iter_count, iter + 1))
                    img_path = image_paths[0][0]
                    visualize_train_img(fig_name, img_path, chirp_amp_curr, confmap_pred[0, :self.n_class, 0, :, :], visual_gt[0, :self.n_class, 0, :, :])
                self.iter_count += 1
            if epoch + 1 in self.config_dict['train_cfg']['eval_epoch_list']:
                self.model.eval()
                maps, ap50s, ap70s, mars = self.eval()
                max_idx, map = max(enumerate(maps), key=lambda x: x[1])
                score = map
                ap50 = ap50s[max_idx]
                ap70 = ap70s[max_idx]
                mar = mars[max_idx]
                self.model.train()
                if score > self.best_score:
                    self.patience_count = 0
                    self.best_map = map
                    self.best_mar = mar
                    self.best_ap50 = ap50
                    self.best_ap70 = ap70
                    self.best_score = score
                    if self.best_score_epoch > 0:
                        os.remove('%s/epoch_%02d_best.pkl' % (self.model_dir, self.best_score_epoch))
                    self.best_score_epoch = epoch + 1
                    print("saving current model ...")
                    if self.ema is not None:
                        status_dict = {
                            'model_name': self.model_name,
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            "ema_999": self.ema.ema.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    else:
                        status_dict = {
                            'model_name': self.model_name,
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    save_model_path = '%s/epoch_%02d_best.pkl' % (self.model_dir, epoch + 1)
                    torch.save(status_dict, save_model_path)
                else:
                    self.patience_count += 1
            if epoch + 1 == self.n_epoch or self.patience_count == self.optim_configs['max_patience']:
                if self.ema is not None:
                    status_dict = {
                        'model_name': self.model_name,
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        "ema_999": self.ema.ema.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                else:
                    status_dict = {
                        'model_name': self.model_name,
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                save_model_path = '%s/epoch_%02d_final.pkl' % (self.model_dir, epoch + 1)
                torch.save(status_dict, save_model_path)
                break
            if 'epoch' in self.schedule_configs['type'] or self.schedule_configs['type'] == 'base-rodnet':
                self.scheduler.step()
        print('Training Finished.')


    def eval(self):
        self.model.eval()
        if self.ema is not None:
            self.ema.ema.eval()
        all_ap50 = []
        all_ap70 = []
        all_map = []
        all_mar = []
        for use_ema_num in range(0, 1):
            with torch.no_grad():
                print("Start eval")
                save_result = False
                test_res_dir = os.path.join(self.model_dir, "temp_results")
                if os.path.exists(test_res_dir):
                    shutil.rmtree(test_res_dir)
                os.mkdir(test_res_dir)
                test_root = os.path.join(self.args.data_dir, "test")
                seq_names = sorted(os.listdir(test_root))
                seq_names = [file.replace('.pkl', '') for file in seq_names]
                for seq_name in seq_names:
                    seq_res_dir = os.path.join(test_res_dir, seq_name)
                    if not os.path.exists(seq_res_dir):
                        os.makedirs(seq_res_dir)
                    seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
                    if not os.path.exists(seq_res_viz_dir):
                        os.makedirs(seq_res_viz_dir)
                    f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
                    f.close()
                for subset in tqdm(seq_names):
                    eval_dataloader = self.eval_dataloader_list[subset]
                    init_genConfmap = ConfmapStack(self.confmap_shape)
                    iter_ = init_genConfmap
                    for i in range(self.train_configs['win_size'] - 1):
                        while iter_.next is not None:
                            iter_ = iter_.next
                        iter_.next = ConfmapStack(self.confmap_shape)
                    for iter, data_dict in enumerate(eval_dataloader):
                        data = data_dict['radar_data']
                        seq_name = data_dict['seq_names'][0]
                        save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
                        start_frame_id = data_dict['start_frame'].item()
                        if self.ema is not None:
                            if use_ema_num == 0:
                                confmap_pred = self.ema.ema(data.float().cuda())
                                ema_num = 0.9990
                        else:
                            confmap_pred = self.model(data.float().cuda())
                        if type(confmap_pred) is list:
                            confmap_pred = confmap_pred[-1]
                        if 'bcel' in self.model_cfg['loss_type'] or 'smooth_l1l' in self.model_cfg['loss_type']:
                            confmap_pred = confmap_pred.sigmoid()
                        confmap_pred = confmap_pred.cpu().detach().numpy()
                        if self.args.use_noise_channel:
                            confmap_pred = confmap_pred[:, :self.n_class, :, :, :]
                        iter_ = init_genConfmap
                        for i in range(confmap_pred.shape[2]):
                            if iter_.next is None and i != confmap_pred.shape[2] - 1:
                                iter_.next = ConfmapStack(self.confmap_shape)
                            iter_.append(confmap_pred[0, :, i, :, :])
                            iter_ = iter_.next
                        for i in range(self.test_configs['test_stride']):
                            res_final = post_process_single_frame(init_genConfmap.confmap, self.dataset, self.config_dict)
                            cur_frame_id = start_frame_id + i
                            write_dets_results_single_frame(res_final, cur_frame_id, save_path, self.dataset)
                            init_genConfmap = init_genConfmap.next
                        if iter == len(eval_dataloader) - 1:
                            offset = self.test_configs['test_stride']
                            cur_frame_id = start_frame_id + offset
                            while init_genConfmap is not None:
                                res_final = post_process_single_frame(init_genConfmap.confmap, self.dataset, self.config_dict)
                                write_dets_results_single_frame(res_final, cur_frame_id, save_path, self.dataset)
                                init_genConfmap = init_genConfmap.next
                                offset += 1
                                cur_frame_id += 1
                        if init_genConfmap is None:
                            init_genConfmap = ConfmapStack(self.confmap_shape)
                olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
                recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
                dataset = CRUW(data_root=self.args.data_dir, sensor_config_name=os.path.join(self.code_dir, 'configs/CRUW/dataset_configs/sensor_config'), object_config_name=os.path.join(self.code_dir, 'configs/CRUW/dataset_configs/object_config'))
                seq_names = sorted(os.listdir(test_res_dir))
                seq_names = [name for name in seq_names if '.' not in name]
                evalImgs_all = []
                n_frames_all = 0
                for seq_name in seq_names:
                    gt_path = os.path.join(self.config_dict['dataset_cfg']['base_root'], 'annotations/test', f'{seq_name}.txt')
                    res_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
                    n_frame = len(os.listdir(os.path.join(self.config_dict['dataset_cfg']['base_root'], 'sequences/test', seq_name, 'IMAGES_0')))
                    evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
                    eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
                    stats = summarize(eval, olsThrs, recThrs, dataset, gl=True)
                    print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90 %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[3] * 100, stats[6] * 100))
                    with open(self.train_log_name, 'a+') as f_log:
                        f_log.write("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90 %.4f \n" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[3] * 100, stats[6] * 100))
                    n_frames_all += n_frame
                    evalImgs_all.extend(evalImgs)

                eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
                stats = summarize(eval, olsThrs, recThrs, dataset, show_sub=[True, self.train_log_name], gl=True)
                print("%s | %.4f | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90 %.4f" % ('Overall'.ljust(18), ema_num, stats[0] * 100, stats[1] * 100, stats[3] * 100, stats[6] * 100))
                with open(self.train_log_name, 'a+') as f_log:
                    f_log.write("%s | %.4f | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90 %.4f \n" % ('Overall'.ljust(18), ema_num, stats[0] * 100, stats[1] * 100, stats[3] * 100, stats[6] * 100))
                if not save_result:
                    shutil.rmtree(test_res_dir)
                all_map.append(stats[0])
                all_ap50.append(stats[1])
                all_ap70.append(stats[3])
                all_mar.append(stats[6])
        self.model.train()
        return all_map, all_ap50, all_ap70, all_mar


def parse_args():
    parser = argparse.ArgumentParser(description='Train in CRUW.')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--data_dir', type=str, help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, help='directory to save trained model')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--code_dir', type=str)
    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)
    set_seed(config_dict['train_cfg']['seed'])
    trainer = Trainer(config_dict, args)
    trainer.train()
