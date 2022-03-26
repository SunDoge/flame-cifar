{
  // 需要调用的 trainer
  _call: 'lib.trainers.of_cifar_sup_trainer.Trainer',

  base_lr: 0.2,
  batch_size: 256,
  num_workers: 8,
  max_epochs: 200,

  base_model: {
    _call: 'lib.models.of_cifar_resnet.cifar_resnet20',
    num_classes: 10,
  },
  optimizer_fn: {
    _use: 'oneflow.optim.SGD',
    momentum: 0.9,
    weight_decay: 1e-4,
    // nesterov: true,
  },
  scheduler_fn: {
    _use: 'oneflow.optim.lr_scheduler.MultiStepLR',
    milestones: [100, 150],
  },

  train_transform: {
    _call: 'lib.transforms.of_cifar_presets.CifarTransform',
    train: true,
  },
  // 这里继承 train_transform, 修改 train = false
  val_transform: $.train_transform {
    train: false,
  },
  train_dataset: {
    _call: 'flowvision.datasets.CIFAR10',
    root: 'data/cifar',
    download: true,
    train: true,
    transform: '$train_transform',  // $train_transform 表示在 python 中，transform = train_transform
  },
  val_dataset: $.train_dataset {
    train: false,
    transform: '$val_transform',
  },
  // 这里使用 flame 提供的帮助函数，会自动包装分布式 data_loader, 并自动使用性能最优参数
  // 比如 pin_memory 和 multiprocess_context
  train_loader: {
    _call: 'flame.oneflow.helpers.create_data_loader',
    dataset: '$train_dataset',
    batch_size: $.batch_size,
    num_workers: $.num_workers,
    shuffle: true,
    drop_last: true,
  },
  val_loader: {
    _call: 'flame.oneflow.helpers.create_data_loader',
    dataset: '$val_dataset',
    batch_size: $.batch_size * 2,
    num_workers: $.num_workers,
    shuffle: false,
    drop_last: false,
  },
}
