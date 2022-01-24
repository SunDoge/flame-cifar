import logging
from typing import Callable, List, Tuple

import torch
from torch import Tensor, nn

from flame.helpers.tensorboard import Rank0SummaryWriter
from flame.pytorch import helpers
from flame.pytorch.metrics.topk_accuracy import topk_accuracy
from flame.pytorch.trainer.trainer_v2 import BaseTrainer
from flame.pytorch.typing_prelude import DataLoader, LrSchedulerFn, OptimizerFn
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
        optimizer_fn: OptimizerFn,
        scheduler_fn: LrSchedulerFn,
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

        model = helpers.create_ddp_model(base_model, args.device)

        # 在 config 中指定为 _use，需要在 python 中填入缺少的参数才能得到实际对象
        optimizer = optimizer_fn(model.parameters(), lr)
        scheduler = scheduler_fn(optimizer)

        # 不一定所有对象都要用 config 生成，而且随时可以加入 config
        criterion = nn.CrossEntropyLoss()

        # rank0 summarywriter 只会在 rank == 0 时写入 tensorboard
        summary_writer = Rank0SummaryWriter(log_dir=args.experiment_dir)

        # 告诉 checkpoint manager, 哪些对象需要写入 checkpoint
        self.state_manager.register_model(base_model)
        self.state_manager.register_optimizer(optimizer)
        self.state_manager.register_lr_scheduler(scheduler)

        self.model = model
        self.base_model = base_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.summary_writer = summary_writer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run(max_epochs)

    def train(self, loader: DataLoader, prefix: str = "train"):
        self.state_manager.train()
        self.set_sampler_epoch(loader)

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

        pm.write_summary(self.summary_writer, self.module_name)

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

        pm.write_summary(self.summary_writer, self.module_name)

    def run(self, max_epochs: int):
        for epoch in self.epoch_range(max_epochs):
            self.train(self.train_loader)
            self.validate(self.val_loader)
            self.scheduler.step()
