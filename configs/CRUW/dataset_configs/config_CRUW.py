dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="/home/jackychou/dataset/CRUW",
    data_root="/home/jackychou/dataset/CRUW/sequences",
    anno_root="/home/jackychou/dataset/CRUW/annotations",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        seqs=['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22',
              '23', '24', '25', '26', '27', '28', '29', '30', '32', '33', '34', '35', '36', '37', '38', '39', '40'],
    ),
    valid=dict(
        subdir='valid',
        seqs=[],
    ),
    test=dict(
        subdir='test',
        seqs=['3', '8', '14', '31'],
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
    }
)

train_cfg = dict(
    batch_size=4,
    win_size=16,
    train_stride=4,
    log_step=50,
    train_step=1,
    seed=2027,
    num_workers=8,
    eval_epoch_list=[i for i in range(6, 56)],
    )

test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
