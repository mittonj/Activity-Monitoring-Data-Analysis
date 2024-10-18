from glob import iglob
from os.path import join as pjoin
from typing import Any, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from module.nets import Batch, device

from torch import as_tensor

# np.random.seed(42)

#
## NOTE: for non-smoothed data
#
# FEATURES = [
#     "time",
#     "sampling_interval",
#     # g-force meter
#     "gFx",
#     "gFy",
#     "gFz",
#     # linear accelerometer
#     "ax",
#     "ay",
#     "az",
#     # gyroscope (rotation)
#     "wx",
#     "wy",
#     "wz",
#     # speed
#     "Speed",
# ]
FEATURES = [
    "time",
    "sampling_interval",
    # g-force meter
    "gFx_smoothed",
    "gFy_smoothed",
    "gFz_smoothed",
    # linear accelerometer
    "ax_smoothed",
    "ax_smoothed",
    "ax_smoothed",
    # gyroscope (rotation)
    "wx_smoothed",
    "wy_smoothed",
    "wz_smoothed",
    # speed
    "Speed",
]
LABEL_2_INT = {"walk": 0, "run": 1, "skate": 2}
INT_2_LABEL = {v: k for k, v in LABEL_2_INT.items()}
N_CLASS = len(LABEL_2_INT)


fname_dir = "data/all_data_v2-oput_60s"
# fname_dir = "data/all_data_v2-small-oput_60s" # XXX

import sys

if len(sys.argv) <= 1:
    print("Need a test data folder as the input")
    sys.exit(0)


# df = pd.concat(map(lambda f: pd.read_json(f, lines = True),
#                    iglob(pjoin(fname_dir, "*.json.gz"))))
df = pd.concat(map(lambda f: pd.read_json(f, lines = True),
                iglob(pjoin(fname_dir, "*.json"))))
df.reset_index(inplace = True, drop = True)


N = 8 # 2 6
T = 512 # 256
C = len(FEATURES)

DATA_TRAIN_RATIO = .6


# | N | batch             |
# | C | channel (feature) |
# | T | seq_len           |


def group_by_n(arr: Sequence[Any], n: int) -> Iterable[Any]:
    n_group = len(arr) // n
    # will drop the remainder
    return [arr[i * n:(i + 1) * n] for i in range(n_group)]

def group_by_n_df(df: pd.DataFrame, n: int) -> Iterable[np.ndarray]:
    np_arr = df[FEATURES].to_numpy(dtype = np.float32)
    n_group = len(df) // n
    # will drop the remainder
    return [np_arr[i * n:(i + 1) * n].T # [T, C] -> [C, T]
            for i in range(n_group)]

# NOTE: train/test/val split
def split_tr_ts() -> Tuple[Any, Any]:
    data = [(i, LABEL_2_INT[df_["activity"].iloc[0]])
            for _, df_ in df.groupby("filename")
            for i in group_by_n_df(df_, T)]

    if (len(data) % N) != 0:
        print(f'[WARN] {len(data)} unable to batch by {N}, '
            'will ignore the remainder')
    np.random.shuffle(data)

    i = int(len(data) * DATA_TRAIN_RATIO)
    return data[:i], data[i:]

def shuffle(d_):
    np.random.shuffle(d_)
    return d_

def shuffle_batching(data) -> Iterable[Batch]:
    np.random.shuffle(data)

    batched = [zip(*g) for g in group_by_n(data, N)]
    return [Batch(iputs = as_tensor(np.array(i)).to(device),
                    targs = as_tensor(o).to(device))
            for i, o in batched]


tr, ts = split_tr_ts()
