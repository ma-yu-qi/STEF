_base_ = [
    '_base_psenet_swin.py',
    '../_base_/datasets/dongba.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(val_interval=40)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[200, 400], end=600),
]

# dataset settings
dongba_textdet_train = _base_.dongba_textdet_train
dongba_textdet_test = _base_.dongba_textdet_test

test_pipeline_ctw = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# pipeline settings
dongba_textdet_train.pipeline = _base_.train_pipeline
dongba_textdet_test.pipeline = test_pipeline_ctw

train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dongba_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dongba_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=64 * 4)

# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5, #每10个epoch保存一次
        max_keep_ckpts=20, #只保留最后一个权重
    ))
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]

