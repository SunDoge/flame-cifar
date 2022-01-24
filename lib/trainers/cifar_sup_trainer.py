import functools
import logging
from typing import Callable, List, Tuple

from torch import Tensor, nn
import torch

from flame import helpers
from flame.pytorch.trainer.trainer_v2 import BaseTrainer
from flame.helpers.tensorboard import Rank0SummaryWriter
from flame.pytorch.metrics.topk_accuracy import topk_accuracy
from lib.args.training_args import Args
from flame.pytorch.typing_prelude import LrScheduler, DataLoader, Optimizer

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
        optimizer_fn: functools.partial,
        scheduler_fn: functools.partial,
        base_lr: float,
        max_epochs: int,
        batch_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ** kwargs
    ) -> None:
        super().__init__(args)

        # 根据 卡数 和 batch size 自动调整 lr
        lr = helpers.scale_lr_linearly(base_lr, batch_size)

        model = helpers.prepare_model(base_model, args.device)

        # 在 config 中指定为 _use，需要在 python 中填入缺少的参数才能得到实际对象
        optimizer: Optimizer = optimizer_fn(model.parameters(), lr=lr)
        scheduler: LrScheduler = scheduler_fn(optimizer)

        # 不一定所有对象都要用 config 生成，而且随时可以加入 config
        criterion = nn.CrossEntropyLoss()

        # rank0 summarywriter 只会在 rank == 0 时写入 tensorboard
        summary_writer = Rank0SummaryWriter(log_dir=args.experiment_dir)

        # 告诉 checkpoint manager, 哪些对象需要写入 checkpoint
        self.state_manager.register_model(base_model)
        self.state_manager.register_optimizer(optimizer)
        self.state_manager.register_lr_scheduler(scheduler)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.summary_writer = summary_writer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # self.run(data_module, max_epochs, debug=args.debug)
        self.run(max_epochs)

    # def forward(self, batch: Tuple[Tensor, Tensor], batch_idx: int, prefix: str):
    #     image, target = batch
    #     image = image.to(self.device, non_blocking=True)
    #     target = target.to(self.device, non_blocking=True)

    #     batch_size = image.size(0)

    #     output = self.model(image)
    #     loss: Tensor = self.criterion(output, target)

    #     # 只有训练阶段才需要 backward 和 优化模型
    #     if self.state.training:
    #         loss.backward()
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))

    #     self.meters.update(prefix, 'acc1', acc1.item(), batch_size)
    #     self.meters.update(prefix, 'acc5', acc5.item(), batch_size)
    #     self.meters.update(prefix, 'loss', loss.item(), batch_size)

    #     if self.every_n_iters(batch_idx, n=self.print_freq, debug=self.args.debug):
    #         # iter_eta 是计时器，会显示当前 iter 和 运行速度
    #         # prefix 是当前的阶段，可以是 train/val/test
    #         _logger.info(f'{self.iter_eta}\t{self.meters[prefix]}')

    #     return batch_size

    # def stage_middleware(self, prefix: str, next_fn: Callable):
    #     # 在 train/val/test 阶段 开始和结束时执行的逻辑

    #     next_fn()

    #     # 将 meters 中记录的数据写入 tensorboard
    #     self.meters.write_summary(
    #         prefix,
    #         self.summary_writer,
    #         self.state.epoch
    #     )

    # def epoch_middleware(self, next_fn: Callable):
    #     # 在 epoch 开始和结束时执行的逻辑

    #     next_fn()

    #     self.scheduler.step()

    #     # meter 可以帮你统计当前值是否最好
    #     helpers.checkpoint_saver.save_checkpoint(
    #         self.checkpoint_manager.state_dict(),
    #         self.args.experiment_dir,
    #         is_best=self.meters['val']['acc1'].is_best(higher_is_better=True)
    #     )

    def train(self, loader, prefix: str = "train"):
        self.state_manager.train()

        pm = self.progress_meter(prefix)

        losses = pm.get('loss')
        acc1 = pm.get('acc1')
        acc5 = pm.get('acc5')

        for batch_idx, batch in pm.enumerate(loader, len(loader)):
            image, target = self.to_device(batch)

            output = self.model(image)
            loss: Tensor = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            top1, top5 = topk_accuracy(output, target, topk=(1, 5))
            batch_size = target.size(0)
            pm.update_batch_size(batch_size)
            losses.update(loss, batch_size)
            acc1.update(top1, batch_size)
            acc5.update(top5, batch_size)

    @torch.inference_mode()
    def validate(self, loader, prefix: str = "val"):
        self.state_manager.eval()

        pm = self.progress_meter(prefix)

        losses = pm.get('loss')
        acc1 = pm.get('acc1')
        acc5 = pm.get('acc5')

        for batch_idx, batch in pm.enumerate(loader, len(loader)):
            image, target = self.to_device(batch)

            output = self.model(image)
            loss: Tensor = self.criterion(output, target)

            top1, top5 = topk_accuracy(output, target, topk=(1, 5))
            batch_size = target.size(0)
            pm.update_batch_size(batch_size)
            losses.update(loss, batch_size)
            acc1.update(top1, batch_size)
            acc5.update(top5, batch_size)

    def run(self, max_epochs: int):
        for epoch in self.epoch_range(max_epochs):
            self.train(self.train_loader)
            self.validate(self.val_loader)
