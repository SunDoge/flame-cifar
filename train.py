import flame
from lib.args.training_args import Args
from flame.pytorch.launcher import run_distributed


@flame.main_fn  # 黑魔法，等价于 if __name__ == '__main__', 帮你少些点代码
def main():
    args = Args.from_args()
    run_distributed(args)
