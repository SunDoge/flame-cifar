# flame cifar

## Init

```shell
git clone https://github.com/SunDoge/flame-cifar.git
git submodule update --init # 因为这里的flame以git submodule的形式引入
```

## Run

### 单卡运行

```shell
python train.py -c configs/cifar10_resnet20_sup.jsonnet --gpu 7 -e exps/000
```

`-e` 为实验名，`flame` 会根据实验名生成实验目录。可以输入`000/pretrain`, `000/finetune` 建立子实验目录。

### 多卡分布式

```shell
python train.py -c configs/cifar10_resnet20_sup.jsonnet --gpu 0-3 -e exps/000
```

`--gpu` 可指定显卡范围，这里等价于 `--gpu 0,1,2,3`, 可以混用 `--gpu 0,1,4-7`。

### Patch

`config` 支持使用 `patch`, 定义在 `configs/patch.libsonnet` 中。如 

- `-a cifar100` 修改 `dataset` 为 `CIFAR100`
- `-a "setep(100)"` 修改训练轮数为 100
- `-a "setbs(256)"` 修改 `batch size` 为 256

## Abbr

sup -> supervised
