import functools
from typing import Tuple, overload
from flame.experimental.trainer import BaseTrainer, DataModule
from lib.args.training_args import Args
from torch import nn, Tensor
from flame import helpers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from flame.helpers.amp import Amp
from flame.pytorch.metrics.topk_accuracy import topk_accuracy
import logging
from typing import Callable
from flame.helpers.tensorboard import Rank0SummaryWriter
_logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def __init__(
        self,
        args: Args,
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

        lr = helpers.scale_lr_linearly(lr, batch_size)

        model = helpers.prepare_model(base_model, args.device)
        optimizer: Optimizer = optimizer_fn(model.parameters(), lr=lr)
        scheduler: LrScheduler = scheduler_fn(optimizer)
        criterion = nn.CrossEntropyLoss()
        summary_writer = Rank0SummaryWriter(log_dir=args.experiment_dir)

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

        if self.state.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))

        self.meters.update(prefix, 'acc1', acc1.item(), batch_size)
        self.meters.update(prefix, 'acc5', acc5.item(), batch_size)
        self.meters.update(prefix, 'loss', loss.item(), batch_size)

        if self.every_n_iters(batch_idx, n=self.print_freq, debug=self.args.debug):
            _logger.info(f'{self.iter_eta}\t{self.meters[prefix]}')

        return batch_size

    def stage_middleware(self, prefix: str, next_fn: Callable):
        next_fn()

        self.meters.write_summary(
            prefix,
            self.summary_writer,
            self.state.epoch
        )

    def epoch_middleware(self, next_fn: Callable):
        next_fn()

        self.scheduler.step()

        helpers.checkpoint_saver.save_checkpoint(
            self.checkpoint_manager.state_dict(),
            self.args.experiment_dir,
            is_best=self.meters['val']['acc1'].is_best()
        )
