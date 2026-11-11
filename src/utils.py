"""统一配置与随机种子入口。"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path=None):
    """默认读取项目根目录的配置，不依赖启动时的工作目录。"""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed, stream="global"):
    """设置 Python/NumPy/PyTorch 种子，返回可供 default_rng 使用的子种子。"""
    # 固定通道编号，避免调用顺序或 Python hash 随机化改变子种子。
    stream_ids = {"global": 0, "generator": 1, "sir": 2, "rl": 3}
    derived_seed = int(
        np.random.SeedSequence([seed, stream_ids[stream]]).generate_state(1)[0]
    )
    random.seed(derived_seed)
    np.random.seed(derived_seed)
    torch.manual_seed(derived_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return derived_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取候选配置并检查随机种子入口")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    print(f"seed={config['seed']}")
    print(f"graph={config['graph']}")
    print(f"hgnn={config['hgnn']}")
    print(f"sir={config['sir']}")
    print(f"rl={config['rl']}")
