# model settings
model = dict(
    type='CSP',
    # pretrained='open-mmlab://msra/hrnetv2_w40',
    pretrained='/root/HR-CSP/pretrained/hrnetv2_w40_imagenet_pretrained.pth',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320)))),
    neck=dict(
        type='CSPNeckGC',
        in_channels=[40, 80, 160, 320],
        out_channels=768,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=1,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CSPRFBHead',
        num_classes=2,
        in_channels=768,
        stacked_convs=1,
        feat_channels=256,
        strides=[4],
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=0.01),
        loss_bbox=dict(type='L1Loss', loss_weight=0.05),
        loss_offset=dict(
            type='SmoothL1Loss', loss_weight=0.1)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.1, #0.2, #0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoCSPORIDataset'
data_root = 'datasets/CityPersons/'
INF = 1e8
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'leftImg8bit_trainvaltest/train.json',
        img_prefix=data_root,

        img_scale=(1280, 640),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        remove_small_box=True,
        small_box_size=8,
        strides=[4],
        regress_ranges=((-1, INF),)),
    val=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        # img_prefix=data_root + 'val_data/',
        # #img_scale=(1333, 800),
        # img_scale = (800, 600),
        # img_norm_cfg=img_norm_cfg,
        # size_divisor=32,
        # flip_ratio=0,
        # with_mask=False,
        # with_crowd=False,
        # with_label=True),
        type=dataset_type,
        ann_file=data_root + '/leftImg8bit_trainvaltest/val_gt_for_mmdetction.json',
        img_prefix=data_root + '/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/leftImg8bit_trainvaltest/val_gt_for_mmdetction.json',
        img_prefix=data_root + '/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.01/10,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
mean_teacher = True
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(mean_teacher = dict(alpha=0.999))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[90])

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 150
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ori_csp3'
load_from = None
# load_from = '/home/ljp/code/mmdetection/work_dirs/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/epoch_22.pth'
resume_from = None
# resume_from = '/home/ljp/code/mmdetection/work_dirs/csp4_mstrain_640_800_x101_64x4d_fpn_gn_2x/epoch_10.pth'
workflow = [('train', 1)]
