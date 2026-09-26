from configs.CRUW.dataset_configs.config_CRUW import *

model_cfg = dict(
    type='D2FA-Net',
    name='D2FA-Net',
    max_dets=20,
    peak_thres=0.15,
    ols_thres=0.3,
    mnet_cfg=(4, 32),
    loss_type='smooth_l1l',
    depths=(2, 2, 4, 2),
    channels=(64, 128, 256, 512),
    fpn_channel=128,
    drop_rate=0.1,
    data_aug=[True, 'old'],
    data_norm=False,
    e_data_aug=False,
    mnet_type=['simDecoupledMnet_avg', True, True],
    tokenMixer=['c', 'c', 'c', 'c'],
    head_num=[1, 1],
    filter_cfg=dict(
        filter_radius=[1, 1, 1, 1],
        filter_type='pre',
        scale=0.25,
    ),
)

train_cfg = dict(
    batch_size=4,
    win_size=16, # 16 / 32
    train_stride=4,
    log_step=50,
    train_step=1,
    seed=2027,
    num_workers=4,
    eval_epoch_list=[i for i in range(1, 26)],
    use_ema=True,
    )

schedule_cfg = dict(
    type='cosine_epoch',
    n_epoch=25,
)

optim_cfg = dict(
    type='adamw',
    lr=0.0003,
    max_patience=5,
)