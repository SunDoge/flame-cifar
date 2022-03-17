import flame
import oneflow.env
from icecream import ic


@flame.main_fn
def main():
    ic(oneflow.env.get_local_rank())
    ic(oneflow.env.get_rank())
