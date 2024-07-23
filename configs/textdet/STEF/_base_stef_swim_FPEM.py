model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.SwinTransformer',
        # embed_dims=96,  # Initial number of features
        # depths=[2, 2, 6, 2],  # Number of layers in each stage
        # num_heads=[3, 6, 12, 24],  # Number of attention heads in different stages
        # window_size=7,  # Size of the window for local self-attention
        # mlp_ratio=4.,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.0,
        # attn_drop_rate=0.0,
        # drop_path_rate=0.2,
        #init_cfg=dict(type='Pretrained', checkpoint='path/to/swin_transformer_checkpoint'),
        ),  # Outputs of which stages to be used, aligning with DBNet head requirements
    neck=dict(
        type='FPEM_FFM',
        in_channels=[96, 192, 384, 768],
        # This needs to be adjusted according to the Swin Transformer configuration
        ),
    det_head=dict(
        type='DBHeadde',
        in_channels=128,
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
