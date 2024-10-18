import pickle
from os import makedirs
from os.path import dirname, isdir, isfile
from typing import Any


def ensure_dir(fpath: str) -> str:
    _p = dirname(fpath)
    if isdir(_p):
        return fpath

    makedirs(_p, exist_ok=True)
    return fpath


def pkl_lod(fpath: str) -> Any:
    if not isfile(fpath):
        return None
    with open(fpath, 'rb') as _f:
        d = pickle.load(_f)
    return d

def pkl_dmp(fpath: str, data: Any) -> None:
    with open(fpath, 'wb') as _f:
        pickle.dump(data, _f, pickle.HIGHEST_PROTOCOL)
    return None
