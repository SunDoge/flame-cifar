import flame
from lib.args.training_args import Args
from flame.distributed_training import start_distributed_training


@flame.main_fn
def main():
    args = Args.from_args()
    start_distributed_training(args)
