model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.MobileNetV2',
        out_indices=(1, 2, 3, 4),  # 选择输出层，需要根据实际MobileNetV2的实现来确定
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        # init_cfg=dict(type='Pretrained', checkpoint='path_to_pretrained_mobilenetv2'),
        ),  # Outputs of which stages to be used, aligning with DBNet head requirements
    neck=dict(
        type='FPEM_FFMese',
        in_channels=[24, 32, 64, 96],
        # This needs to be adjusted according to the Swin Transformer configuration
        ),
    det_head=dict(
        type='DBHeadde',
        in_channels=512,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
