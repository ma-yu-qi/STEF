_base_ = [
    '_base_stef_swim_FPEMese.py',
    '../_base_/datasets/dongba.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# dataset settings
dongba_textdet_train = _base_.dongba_textdet_train
dongba_textdet_train.pipeline = _base_.train_pipeline
dongba_textdet_test = _base_.dongba_textdet_test
dongba_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dongba_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dongba_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=2)

# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4, #每10个epoch保存一次
        max_keep_ckpts=20, #只保留最后一个权重
    ))
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=4)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]
