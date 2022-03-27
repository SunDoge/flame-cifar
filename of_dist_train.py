"""
python -m oneflow.distributed.launch --nproc_per_node 2 of_dist_train.py
"""

import os

# from lib.models.of_cifar_resnet import cifar_resnet20
import oneflow as flow
from oneflow import nn
import oneflow.env

S0 = flow.sbp.split(0)


class TrainGraph(nn.Graph):

    def __init__(self, model, optimizer, criterion):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        return y_pred, loss


def to_local(tensor, pure_local=True):
    if tensor.is_global:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_global(sbp=flow.sbp.broadcast).to_local()

    return tensor


def main():
    device_id = int(os.environ.get('LOCAL_RANK'))
    device = flow.device('cuda:{}'.format(device_id))
    print(device)
    # model = cifar_resnet20(num_classes=10)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 10)
    )

    placement = oneflow.env.all_device_placement("cuda")
    print(placement)
    model = model.to_global(
        placement=placement, sbp=flow.sbp.broadcast,
    )
    # ddp_model = nn.parallel.DistributedDataParallel(model)
    ddp_model = model
    # It's fine when momentum=0
    optimizer = flow.optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_graph = TrainGraph(ddp_model, optimizer, criterion)
    # train_graph.debug(1)

    x = flow.rand(2, 3, 32, 32).to_global(
        placement=placement,
        sbp=S0
    )
    y = flow.randint(0, 10, (2,)).to_global(
        placement=placement,
        sbp=S0
    )

    print(y.shape)
    print(to_local(y))

    print('*' * 50)
    for _ in range(10):
        output, loss = train_graph(x, y)

        print(to_local(loss))

    print('*' * 50)
    for _ in range(10):
        output, loss = train_graph(x, y)

        print(to_local(loss))


if __name__ == '__main__':
    main()
