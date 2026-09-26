from configs.CRUW.dataset_configs.config_CRUW import *

model_cfg = dict(
    type='STCT',
    name='STCT',
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    mnet_cfg=(4, 32),
    loss_type='bcel',
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
    use_ema=True
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