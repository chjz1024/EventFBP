import jax
import jax.numpy as jnp


def rmse(err: jax.Array, ts: jax.Array = None):
    if ts is None:
        return (err**2).mean() ** 0.5
    else:
        return (jnp.trapezoid(err**2, ts, axis=0) / (ts[-1] - ts[0])) ** 0.5


def drotang(rot1, rot2):
    return jnp.linalg.norm((rot1.inv() * rot2).as_rotvec(), axis=1) * 180 / jnp.pi
