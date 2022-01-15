import functools
import logging
from typing import Callable, Tuple

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler

from flame import helpers
from flame.experimental.trainer import BaseTrainer, DataModule
from flame.helpers.tensorboard import Rank0SummaryWriter
from flame.pytorch.metrics.topk_accuracy import topk_accuracy
from lib.args.training_args import Args

_logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    # 参数对应 config 中的配置, 除了 args，config 为默认参数，分别对应 命令行参数 和 配置文件
    # _call 会自动调用并得到对象
    # _use 会得到一个函数，执行之后才能拿到对象
    # 我习惯写类型标注，类型标注可以省略
    def __init__(
        self,
        args: Args,
        # config: dict,
        base_model: nn.Module,
        data_module: DataModule,
        optimizer_fn: functools.partial,
        scheduler_fn: functools.partial,
        lr: float,
        max_epochs: int,
        batch_size: int,
        print_freq: int,
        ** kwargs
    ) -> None:
        super().__init__()

        # 根据 卡数 和 batch size 自动调整 lr
        lr = helpers.scale_lr_linearly(lr, batch_size)

        model = helpers.prepare_model(base_model, args.device)

        # 在 config 中指定为 _use，需要在 python 中填入缺少的参数才能得到实际对象
        optimizer: Optimizer = optimizer_fn(model.parameters(), lr=lr)
        scheduler: LrScheduler = scheduler_fn(optimizer)

        # 不一定所有对象都要用 config 生成，而且随时可以加入 config
        criterion = nn.CrossEntropyLoss()

        # rank0 summarywriter 只会在 rank == 0 时写入 tensorboard
        summary_writer = Rank0SummaryWriter(log_dir=args.experiment_dir)

        # 告诉 checkpoint manager, 哪些对象需要写入 checkpoint
        self.checkpoint_manager.register_model(base_model)
        self.checkpoint_manager.register_optimizer(optimizer)
        self.checkpoint_manager.register_lr_scheduler(scheduler)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = args.device
        self.summary_writer = summary_writer
        self.args = args
        self.print_freq = print_freq

        self.run(data_module, max_epochs, debug=args.debug)

    def forward(self, batch: Tuple[Tensor, Tensor], batch_idx: int, prefix: str):
        image, target = batch
        image = image.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        batch_size = image.size(0)

        output = self.model(image)
        loss: Tensor = self.criterion(output, target)

        # 只有训练阶段才需要 backward 和 优化模型
        if self.state.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))

        self.meters.update(prefix, 'acc1', acc1.item(), batch_size)
        self.meters.update(prefix, 'acc5', acc5.item(), batch_size)
        self.meters.update(prefix, 'loss', loss.item(), batch_size)

        if self.every_n_iters(batch_idx, n=self.print_freq, debug=self.args.debug):
            # iter_eta 是计时器，会显示当前 iter 和 运行速度
            # prefix 是当前的阶段，可以是 train/val/test
            _logger.info(f'{self.iter_eta}\t{self.meters[prefix]}')

        return batch_size

    def stage_middleware(self, prefix: str, next_fn: Callable):
        # 在 train/val/test 阶段 开始和结束时执行的逻辑

        next_fn()

        # 将 meters 中记录的数据写入 tensorboard
        self.meters.write_summary(
            prefix,
            self.summary_writer,
            self.state.epoch
        )

    def epoch_middleware(self, next_fn: Callable):
        # 在 epoch 开始和结束时执行的逻辑

        next_fn()

        self.scheduler.step()

        # meter 可以帮你统计当前值是否最好
        helpers.checkpoint_saver.save_checkpoint(
            self.checkpoint_manager.state_dict(),
            self.args.experiment_dir,
            is_best=self.meters['val']['acc1'].is_best(higher_is_better=True)
        )
