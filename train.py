import flame
from flame.distributed_training import start_distributed_training
from lib.args.training_args import Args


@flame.main_fn  # 黑魔法，等价于 if __name__ == '__main__', 帮你少些点代码
def main():
    args = Args.from_args()
    start_distributed_training(args)
