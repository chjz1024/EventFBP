import jax
import numpy as np
import torch
from jax.scipy.spatial.transform import Rotation


def warp3drot_linapprox(
    rotvec: jax.Array | torch.Tensor | np.ndarray,
    txy_array: jax.Array | torch.Tensor | np.ndarray,
    t0: float,
):
    """
    With small angle assumption
    R = I + [rot]_x
    txy_array: [Nx3]
    t0: [1]
    rotvec: [3]
    """
    linalgmod = (
        torch if isinstance(rotvec, torch.Tensor) else rotvec.__array_namespace__()
    )
    rot = rotvec * (t0 - txy_array[:, :1])
    coords = linalgmod.hstack((txy_array[:, 1:], linalgmod.ones_like(txy_array[:, :1])))
    coords = coords + linalgmod.linalg.cross(rot, coords)
    return coords[:, :2] / coords[:, 2:]


def warp3dtrans(velvec: jax.Array, txy_array: jax.Array, t0: float):
    linalgmod = (
        torch if isinstance(velvec, torch.Tensor) else velvec.__array_namespace__()
    )
    trans = velvec * (t0 - txy_array[:, :1])
    coords = (
        linalgmod.hstack((txy_array[:, 1:], linalgmod.ones_like(txy_array[:, :1])))
        + trans
    )
    return coords[:, :2] / coords[:, 2:]


def warp6dof_linapprox(
    velrotvec: jax.Array | torch.Tensor | np.ndarray,
    txyz_array: jax.Array | torch.Tensor | np.ndarray,
    t0: float,
):
    linalgmod = (
        torch
        if isinstance(velrotvec, torch.Tensor)
        else velrotvec.__array_namespace__()
    )
    transform = velrotvec * (t0 - txyz_array[:, :1])
    coords = (
        txyz_array[:, 1:]
        + transform[:, :3]
        + linalgmod.linalg.cross(transform[:, 3:], txyz_array[:, 1:])
    )
    return coords[:, :2] / coords[:, 2:]


def warp3drot_exact(
    rotvec: jax.Array | torch.Tensor | np.ndarray,
    txy_array: jax.Array | torch.Tensor | np.ndarray,
    t0: float,
):
    """
    Exact Rodrigues' rotation formula
    v_rot = v + sin(theta)(e x v) + (1-cos(theta))(e x (e x v))
    txy_array: [Nx3]
    t0: [1]
    rotvec: [3]
    """
    linalgmod = (
        torch if isinstance(rotvec, torch.Tensor) else rotvec.__array_namespace__()
    )
    theta = linalgmod.linalg.norm(rotvec)
    e = rotvec / theta
    angle = (t0 - txy_array[:, :1]) * theta
    coords = linalgmod.hstack((txy_array[:, 1:], linalgmod.ones_like(txy_array[:, :1])))
    coords = (
        coords
        + linalgmod.sin(angle) * linalgmod.linalg.cross(e, coords)
        + (1 - linalgmod.cos(angle))
        * linalgmod.linalg.cross(e, linalgmod.linalg.cross(e, coords))
    )
    return coords[:, :2] / coords[:, 2:]