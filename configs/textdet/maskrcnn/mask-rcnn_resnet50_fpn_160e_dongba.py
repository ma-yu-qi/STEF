_base_ = [
    '_base_mask-rcnn_resnet50_fpn.py',
    '../_base_/datasets/dongba.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.001))
train_cfg = dict(max_epochs=20)
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=20, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=10),
]

# dataset settings
dongba_textdet_train = _base_.dongba_textdet_train
dongba_textdet_test = _base_.dongba_textdet_test

# test pipeline for CTW1500
ctw_test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1600, 1600), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

dongba_textdet_train.pipeline = _base_.train_pipeline
dongba_textdet_test.pipeline = ctw_test_pipeline

train_dataloader = dict(
    batch_size=10,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dongba_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dongba_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10, #每10个epoch保存一次
        max_keep_ckpts=20, #只保留最后多少次权重
    ))
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=10)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]
