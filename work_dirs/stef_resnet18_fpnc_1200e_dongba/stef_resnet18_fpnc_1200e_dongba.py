auto_scale_lr = dict(base_batch_size=2)
default_hooks = dict(
    checkpoint=dict(interval=4, max_keep_ckpts=20, type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dongba_textdet_data_root = 'data/dongba'
dongba_textdet_test = dict(
    ann_file='textdet_test.json',
    data_root='data/dongba',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            1333,
            736,
        ), type='Resize'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    test_mode=True,
    type='OCRDataset')
dongba_textdet_train = dict(
    ann_file='textdet_train.json',
    data_root='data/dongba',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            brightness=0.12549019607843137,
            op='ColorJitter',
            saturation=0.5,
            type='TorchVisionWrapper'),
        dict(
            args=[
                [
                    'Fliplr',
                    0.5,
                ],
                dict(cls='Affine', rotate=[
                    -10,
                    10,
                ]),
                [
                    'Resize',
                    [
                        0.5,
                        3.0,
                    ],
                ],
            ],
            type='ImgAugWrapper'),
        dict(min_side_ratio=0.1, type='RandomCrop'),
        dict(keep_ratio=True, scale=(
            640,
            640,
        ), type='Resize'),
        dict(size=(
            640,
            640,
        ), type='Pad'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
            ),
            type='PackTextDetInputs'),
    ],
    type='OCRDataset')
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(type='mmdet.SwinTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=512,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(text_repr_type='quad', type='DBPostprocessor'),
        type='DBHeadde'),
    neck=dict(in_channels=[
        96,
        192,
        384,
        768,
    ], type='FPEM_FFMese'),
    type='DBNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='Adam'), type='OptimWrapper')
param_scheduler = [
    dict(factor=1.0, type='ConstantLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/dongba',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                736,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        736,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='data/dongba',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                brightness=0.12549019607843137,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
            dict(
                args=[
                    [
                        'Fliplr',
                        0.5,
                    ],
                    dict(cls='Affine', rotate=[
                        -10,
                        10,
                    ]),
                    [
                        'Resize',
                        [
                            0.5,
                            3.0,
                        ],
                    ],
                ],
                type='ImgAugWrapper'),
            dict(min_side_ratio=0.1, type='RandomCrop'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(size=(
                640,
                640,
            ), type='Pad'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                ),
                type='PackTextDetInputs'),
        ],
        type='OCRDataset'),
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        brightness=0.12549019607843137,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        args=[
            [
                'Fliplr',
                0.5,
            ],
            dict(cls='Affine', rotate=[
                -10,
                10,
            ]),
            [
                'Resize',
                [
                    0.5,
                    3.0,
                ],
            ],
        ],
        type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/dongba',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                736,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=3,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/stef_resnet18_fpnc_1200e_dongba'
