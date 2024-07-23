_base_ = [
    '../_base_/datasets/dongba.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_panet_resnet18_fpem-ffm.py',
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20), )

# dataset settings
dongba_textdet_train = _base_.dongba_textdet_train
dongba_textdet_test = _base_.dongba_textdet_test
# pipeline settings
dongba_textdet_train.pipeline = _base_.train_pipeline
dongba_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dongba_textdet_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dongba_textdet_test)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64)

# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3, #每10个epoch保存一次
        max_keep_ckpts=30, #只保留最后一个权重
    ))
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=3)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]
