import numpy as np
import torch


def constructStructuredEvarray(t, x, y, p, w=None):
    if w is None:
        data = np.empty(
            len(t),
            dtype=np.dtype(
                [("t", t.dtype), ("x", x.dtype), ("y", y.dtype), ("p", "i2")]
            ),
        )
        data["t"] = t
        data["x"] = x
        data["y"] = y
        data["p"] = p
    else:
        data = np.empty(
            len(t),
            dtype=np.dtype(
                [
                    ("t", t.dtype),
                    ("x", x.dtype),
                    ("y", y.dtype),
                    ("p", "i2"),
                    ("w", w.dtype),
                ]
            ),
        )
        data["t"] = t
        data["x"] = x
        data["y"] = y
        data["p"] = p
        data["w"] = w
    return data


def evarray2txyparray(evarray):
    return np.c_[evarray["t"], evarray["x"], evarray["y"], evarray["p"]]


def evarray2txyptensor(evarray, dtype=torch.float32):
    return torch.tensor(
        np.c_[evarray["t"], evarray["x"], evarray["y"], evarray["p"]], dtype=dtype
    )


def txyparray2evarray(txyp_array):
    return constructStructuredEvarray(
        txyp_array[:, 0], txyp_array[:, 1], txyp_array[:, 2], txyp_array[:, 3]
    )
