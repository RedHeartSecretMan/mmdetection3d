auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
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
input_modality = dict(use_camera=True, use_lidar=False)
load_from = 'work_dirs/fcos3d_nus/latest.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=101,
        frozen_stages=1,
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
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='caffe',
        type='mmdet.ResNet'),
    bbox_head=dict(
        attr_branch=(256, ),
        bbox_coder=dict(code_size=9, type='FCOS3DBBoxCoder'),
        center_sampling=True,
        centerness_on_reg=True,
        cls_branch=(256, ),
        conv_bias=True,
        dcn_on_last_conv=True,
        diff_rad_by_sin=True,
        dir_branch=(256, ),
        dir_limit_offset=0,
        dir_offset=0.7854,
        feat_channels=256,
        group_reg_dims=(
            2,
            1,
            3,
            1,
            2,
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
        loss_dir=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        norm_on_bbox=True,
        num_classes=10,
        pred_attrs=True,
        pred_velo=True,
        reg_branch=(
            (256, ),
            (256, ),
            (256, ),
            (256, ),
            (),
        ),
        stacked_convs=2,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='FCOSMono3DHead',
        use_direction_classifier=True),
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
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='mmdet.FPN'),
    test_cfg=dict(
        max_per_img=200,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
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
            0.05,
            0.05,
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
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts=''),
        data_root='data/nuscenes/',
        load_type='mv_image_based',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
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
        type='NuScenesDataset',
        use_valid_flag=True),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(scale_factor=1.0, type='mmdet.Resize'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts=''),
        data_root='data/nuscenes/',
        load_type='mv_image_based',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=True,
                with_bbox=True,
                with_bbox_3d=True,
                with_bbox_depth=True,
                with_label=True,
                with_label_3d=True),
            dict(keep_ratio=True, scale=(
                1600,
                900,
            ), type='mmdet.Resize'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'attr_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset',
        use_valid_flag=True),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=True,
        with_bbox=True,
        with_bbox_3d=True,
        with_bbox_depth=True,
        with_label=True,
        with_label_3d=True),
    dict(keep_ratio=True, scale=(
        1600,
        900,
    ), type='mmdet.Resize'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'attr_labels',
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
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts=''),
        data_root='data/nuscenes/',
        load_type='mv_image_based',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
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
        type='NuScenesDataset',
        use_valid_flag=True),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
