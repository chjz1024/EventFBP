import jax
import jax.numpy as jnp

# from jax.scipy.spatial.transform import Rotation, Slerp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from functools import partial

"""
Unit quaternions for rotation representation
"""


@jax.jit
@partial(jnp.vectorize, signature="(4)->(3)")
def quat_from_rotvec(rotvec):
    # assert rotvec.shape == (3,)
    theta = jnp.linalg.norm(rotvec)
    return jnp.r_[jnp.sinc(theta / (2 * jnp.pi)) * rotvec / 2, jnp.cos(theta / 2)]


@jax.jit
@partial(jnp.vectorize, signature="(3)->(4)")
def rotvec_from_quat(quat):
    """
    quat: xyzw
    """
    # assert quat.shape == (4,)
    quat = quat / jnp.linalg.norm(quat)
    return 2 * quat[:3] / jnp.sinc(jnp.acos(quat[3]) / jnp.pi)


@jax.jit
@partial(jnp.vectorize, signature="(4),(4)->(4)")
def qmul(lhs, rhs):
    # assert lhs.shape == (4,) and rhs.shape == (4,)
    return jnp.r_[
        lhs[3] * rhs[:3] + rhs[3] * lhs[:3] + jnp.cross(lhs[:3], rhs[:3]),
        lhs[3] * rhs[3] - lhs[:3] @ rhs[:3],
    ]


@jax.jit
def qmul_stateful(lhs, rhs):
    out = qmul(lhs, rhs)
    return out, out


@jax.jit
def qcumprod(quats, q0=jnp.array([0.0, 0.0, 0.0, 1.0])):
    return jax.lax.scan(qmul_stateful, q0, quats)[1]


@jax.jit
def dqmul_stateful(lhs, rhs):
    q0, dq0 = lhs[:4], lhs[4:]
    q1, dq1 = rhs[:4], rhs[4:]
    out = jnp.r_[qmul(q0, q1), qmul(q0, dq1) + qmul(q1, dq0)]
    return out, out


@jax.custom_jvp
def qcumprod_test(quats, q0=jnp.array([0.0, 0.0, 0.0, 1.0])):
    return qcumprod(quats, q0)


@qcumprod_test.defjvp
def qcumprod_jvp(primals, tangents):  # unable to perform linearization
    (quats, q0) = primals
    (dquats, dq0) = tangents
    ret = jax.lax.scan(
        dqmul_stateful,
        jnp.r_[q0, dq0],
        jnp.c_[quats, dquats],
    )[1]

    return ret[:, :4], ret[:, 4:]


@jax.jit
def qprod(quats, q0=jnp.array([0.0, 0.0, 0.0, 1.0])):
    return jax.lax.scan(qmul_stateful, q0, quats)[0]


@jax.jit
@partial(jnp.vectorize, signature="(4),(3)->(3)")
def qapply(quat, pos):
    return qmul(qmul(quat, jnp.r_[pos, 0]), qinv(quat))[:3]


@jax.jit
@partial(jnp.vectorize, signature="(4)->(4)")
def qexp(quat):
    vnorm = jnp.linalg.norm(quat[:3])
    return (
        jnp.exp(quat[3]) * jnp.r_[quat[:3] * jnp.sinc(vnorm / jnp.pi), jnp.cos(vnorm)]
    )


@jax.jit
@partial(jnp.vectorize, signature="(4)->(4)")
def qlog(quat):
    qnorm, vnorm = jnp.linalg.norm(quat), jnp.linalg.norm(quat[:3])
    return jnp.r_[quat[:3] * jnp.acos(quat[3] / qnorm) / vnorm, jnp.log(qnorm)]


@jax.jit
@partial(jnp.vectorize, signature="(4),(1)->(4)")
def qpow(quat, exponent):
    return qexp(exponent * qlog(quat))


@jax.jit
@partial(jnp.vectorize, signature="(4)->(4)")
def qinv(quat):
    # return qconj(quat) / jnp.linalg.norm(quat) ** 2
    return qconj(quat) / (quat**2).sum()


@jax.jit
@partial(jnp.vectorize, signature="(4),(4)->(4)")
def qdiv(lhs, rhs):
    """
    q0 * q1^-1
    """
    return qmul(lhs, qinv(rhs))


@jax.jit
@partial(jnp.vectorize, signature="(4)->(4)")
def qconj(quat):
    return jnp.r_[-quat[:3], quat[3]]


@jax.jit
@partial(jnp.vectorize, signature="(4),(4),(1)->(4)")
def slerp(q0, q1, t):
    # return qmul(qpow(qmul(q1, qinv(q0)), t), q0)
    # Assuming Unit Quaternion
    q1dq0 = qmul(jnp.sign(q0 @ q1) * q1, qinv(q0))  # dq = q1 * q0^-1
    phi = jnp.acos(q1dq0[3])
    nhat = q1dq0[:3] / jnp.linalg.norm(q1dq0[:3])
    dqpt = jnp.r_[nhat * jnp.sin(t * phi), jnp.cos(t * phi)]
    return qmul(dqpt, q0)


@jax.jit
def cslerp(times, timestamps, quaternions):
    indices = jnp.searchsorted(timestamps, times)
    t = (times - timestamps[indices - 1]) / (
        timestamps[indices] - timestamps[indices - 1]
    )
    q0 = quaternions[indices - 1]
    q1 = quaternions[indices]
    return slerp(q0, q1, t[:, None])


# class Slerp:
#     def __init__(self, timestamps, quaternions):
#         assert (timestamps[1:] > timestamps[:-1]).all()
#         self.timestamps = timestamps
#         self.quaternions = quaternions

#     def __call__(self, times):
#         indices = jnp.searchsorted(self.timestamps, times)
#         t = (times - self.timestamps[indices - 1]) / (
#             self.timestamps[indices] - self.timestamps[indices - 1]
#         )
#         q0 = self.quaternions[indices - 1]
#         q1 = self.quaternions[indices]
#         return vslerp(q0, q1, t)


class seplerp:
    def __init__(self, timestamps, translations, quaternions):
        self.lerp = interp1d(timestamps, translations, kind="linear", axis=1)
        self.slerp = Slerp(timestamps, Rotation.from_quat(quaternions))

    def __call__(self, times):
        trans = self.lerp(times)
        rots = self.slerp(times)
        return trans, rots


class sclerp:
    def __init__(self, timestamps, translations, quaternions):
        pass

    def __call__(self, times):
        pass
