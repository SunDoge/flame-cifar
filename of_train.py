import flame
import oneflow.env
from icecream import ic
from lib.args.of_training_args import Args
from flame.oneflow.launcher import run_distributed

@flame.main_fn
def main():
    args = Args.from_args()
    ic(oneflow.env.get_rank())
    ic(oneflow.env.get_local_rank())

    run_distributed(args)
