import flame
from flame.distributed_training import start_distributed_training
from lib.args.training_args import Args
from flame.config_parser import ConfigParser2
import logging
from flame.core.logging import TqdmHandler
from flame.pytorch.helpers import cudnn_benchmark_if_possible


@flame.main_fn  # 黑魔法，等价于 if __name__ == '__main__', 帮你少些点代码
def main():
    logging.basicConfig(level=logging.DEBUG, handlers=[TqdmHandler()])

    args = Args.from_args()

    args.init_process_group_from_file(0)
    # start_distributed_training(args)
    cudnn_benchmark_if_possible()
    ConfigParser2(args=args).parse_root_config(args.config)
