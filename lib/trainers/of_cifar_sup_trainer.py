from lib.args.of_training_args import Args
from oneflow.utils.data import DataLoader

class Trainer:

    def __init__(
        self,
        args: Args,
        train_loader: DataLoader,
        val_loader: DataLoader,
        base_model,
        **kwargs
    ) -> None:
        pass
