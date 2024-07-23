model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.Res2Net',
        depth=50,  # 选择Res2Net的版本，如Res2Net50
        scales=4,  # scale维度，通常为4
        base_width=26,  # 基础宽度，Res2Net的一个参数
        num_stages=4,  # 残差阶段数，通常与标准ResNet保持一致
        out_indices=(0, 1, 2, 3),  # 输出的阶段，这决定了哪些阶段的输出被用于后续处理
        frozen_stages=-1,  # 冻结阶段，-1表示不冻结任何阶段
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化配置
        norm_eval=False,  # 在训练时不更新BN的统计参数
        style='pytorch',  # ResNet的实现风格，'pytorch'是常见选项
        ),  # Outputs of which stages to be used, aligning with DBNet head requirements
    neck=dict(
        type='FPEM_FFMese',
        in_channels=[256, 512, 1024, 2048],
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
