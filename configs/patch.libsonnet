{
  setlr(lr):: {
    lr: lr,
  },
  setep(epochs):: {
    max_epochs: epochs,
  },
  cifar100: {
    local this = self,
    call: 'torchvision.datasets.CIFAR100',
    train_dataset+: {
      _call: this.call,
    },
    val_dataset+: {
      _call: this.call,
    },
    base_model+: {
      num_classes: 100,
    },
  },
}
