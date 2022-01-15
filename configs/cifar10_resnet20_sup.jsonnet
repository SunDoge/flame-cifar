{
  _call: 'lib.trainers.cifar_sup_trainer.Trainer',
  lr: 0.2,
  batch_size: 128,
  num_workers: 4,
  max_epochs: 200,
  print_freq: 50,

  base_model: {
    _call: 'lib.models.cifar_resnet.cifar_resnet20',
    num_classes: 10,
  },
  optimizer_fn: {
    _use: 'torch.optim.SGD',
    momentum: 0.9,
    weight_decay: 1e-4,
    nesterov: true,
  },
  scheduler_fn: {
    _use: 'torch.optim.lr_scheduler.MultiStepLR',
    milestones: [100, 150],
  },

  train_transform: {
    _call: 'lib.transforms.cifar_presets.CifarTransform',
    train: true,
  },
  val_transform: $.train_transform {
    train: false,
  },
  train_dataset: {
    _call: 'torchvision.datasets.CIFAR10',
    root: 'data/cifar10',
    download: true,
    train: true,
    transform: '$train_transform',
  },
  val_dataset: $.train_dataset {
    train: false,
    transform: '$val_transform',
  },
  train_loader: {
    _call: 'flame.helpers.create_data_loader',
    dataset: '$train_dataset',
    batch_size: $.batch_size,
    num_workers: $.num_workers,
  },
  val_loader: $.train_loader {
    dataset: '$val_dataset',
    batch_size: $.batch_size * 2,
  },
  data_module: {
    _call: 'flame.experimental.trainer.DataModule',
    train_loader: '$train_loader',
    val_loader: '$val_loader',
  },
}
