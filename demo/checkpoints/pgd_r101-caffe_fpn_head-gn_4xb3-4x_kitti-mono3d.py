auto_scale_lr = dict(base_batch_size=12, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=True, use_lidar=False)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet101_caffe',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='mmdet.ResNet'),
    bbox_head=dict(
        attr_branch=(256, ),
        bbox_code_size=7,
        bbox_coder=dict(
            base_depths=((
                28.01,
                16.32,
            ), ),
            base_dims=(
                (
                    0.8,
                    1.73,
                    0.6,
                ),
                (
                    1.76,
                    1.73,
                    0.6,
                ),
                (
                    3.9,
                    1.56,
                    1.6,
                ),
            ),
            code_size=7,
            type='PGDBBoxCoder'),
        center_sampling=True,
        centerness_branch=(256, ),
        centerness_on_reg=True,
        cls_branch=(256, ),
        conv_bias=True,
        dcn_on_last_conv=True,
        depth_bins=8,
        depth_branch=(256, ),
        depth_range=(
            0,
            70,
        ),
        depth_unit=10,
        diff_rad_by_sin=True,
        dir_branch=(256, ),
        dir_offset=0.7854,
        division='uniform',
        feat_channels=256,
        group_reg_dims=(
            2,
            1,
            3,
            1,
            16,
            4,
        ),
        in_channels=256,
        loss_attr=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_centerness=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_depth=dict(
            alpha=1.0, beta=3.0, loss_weight=1.0,
            type='UncertainSmoothL1Loss'),
        loss_dir=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        norm_on_bbox=True,
        num_classes=3,
        pred_attrs=False,
        pred_bbox2d=True,
        pred_keypoints=True,
        pred_velo=False,
        reg_branch=(
            (256, ),
            (256, ),
            (256, ),
            (256, ),
            (256, ),
            (256, ),
        ),
        regress_ranges=(
            (
                -1,
                64,
            ),
            (
                64,
                128,
            ),
            (
                128,
                256,
            ),
            (
                256,
                100000000.0,
            ),
        ),
        stacked_convs=2,
        strides=(
            4,
            8,
            16,
            32,
        ),
        type='PGDHead',
        use_depth_classifier=True,
        use_direction_classifier=True,
        use_onlyreg_proj=True,
        weight_dim=1),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='Det3DDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='mmdet.FPN'),
    test_cfg=dict(
        max_per_img=20,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.05,
        score_thr=0.001,
        use_rotate_nms=True),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        debug=False,
        pos_weight=-1),
    type='FCOSMono3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=48,
        gamma=0.1,
        milestones=[
            32,
            44,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(scale_factor=1.0, type='mmdet.Resize'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(scale_factor=1.0, type='mmdet.Resize'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=48, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='kitti_infos_train.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox=True,
                with_bbox_3d=True,
                with_bbox_depth=True,
                with_label=True,
                with_label_3d=True),
            dict(keep_ratio=True, scale=(
                1242,
                375,
            ), type='mmdet.Resize'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox=True,
        with_bbox_3d=True,
        with_bbox_depth=True,
        with_label=True,
        with_label_3d=True),
    dict(keep_ratio=True, scale=(
        1242,
        375,
    ), type='mmdet.Resize'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'centers_2d',
            'depths',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root='data/kitti/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(scale_factor=1.0, type='mmdet.Resize'),
            dict(keys=[
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
