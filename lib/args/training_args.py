from dataclasses import dataclass
from typing import Optional
from flame.arguments import BaseArgs
import typed_args as ta


@dataclass
class Args(BaseArgs):

    # 可以在默认命令行参数的基础上，增加自己的命令行参数
    resume: Optional[str] = ta.add_argument(
        '--resume', help='恢复训练的checkpoint路径'
    )
