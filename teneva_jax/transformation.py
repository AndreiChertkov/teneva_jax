"""Package teneva, module transformation: transformation of TT-tensors.

This module contains the function for transformation of the TT-tensor into full
(numpy) format.

"""
import jax
import jax.numpy as jnp


def full(Y):
    """Export TT-tensor to the full (numpy) format.

    Args:
        Y (list): TT-tensor.

    Returns:
        jnp.ndarray: multidimensional array related to the given TT-tensor.

    Note:
         This function can only be used for relatively small tensors, because
         the resulting tensor will have n^d elements and may not fit in memory
         for large dimensions. And his function does not take advantage of
         jax's ability to speed up the code and can be slow, but it should only
         be meaningfully used for tensors of small dimensions.

    """
    Yl, Ym, Yr = Y

    Z = Yl[0, :, :]
    for G in Ym:
        Z = jnp.tensordot(Z, G, 1)
    Z = jnp.tensordot(Z, Yr[:, :, 0], 1)

    return Z


def orthogonalize_rtl(Y):
    """Orthogonalization for TT-tensor from right to left.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        list: TT-tensor with right orthogonalized modes.

    Note:
        It works now only for TT-tensors with mode size greater than TT-rank.

    """
    def body(R, G):
        r, n = G.shape[:2]

        G = jnp.reshape(G, (r*n, -1), order='F')
        G = jnp.reshape(G @ R, (r, n, -1), order='F')

        G = jnp.reshape(G, (r, -1), order='F')
        Q, R = jnp.linalg.qr(G.T, mode='reduced')
        G = jnp.reshape(Q.T, (r, n, -1), order='F')

        return R.T, G

    Yl, Ym, Yr = Y
    r, n = Yr.shape[:2]

    R, Yr = body(jnp.ones((1, 1)), Yr)
    R, Ym = jax.lax.scan(body, R, Ym, reverse=True)

    Yl = jnp.reshape(Yl, (n, r), order='F')
    Yl = Yl @ R
    Yl = jnp.reshape(Yl, (1, n, r), order='F')

    return [Yl, Ym, Yr]


def orthogonalize_rtl_stab(Y):
    """Orthogonalization for TT-tensor from right to left with stab. factor.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        (list, jnp.ndarray): the scaled TT-tensor Y with right orthogonalized
        modes and stabilization factor p for each TT-core (array of the length
        d). The resulting tensor is Y * 2^{sum(p)}.

    Note:
        It works now only for TT-tensors with mode size greater than TT-rank.

    """
    def body(R, G):
        r, n = G.shape[:2]

        G = jnp.reshape(G, (r*n, -1), order='F')
        G = jnp.reshape(G @ R, (r, n, -1), order='F')

        G = jnp.reshape(G, (r, -1), order='F')
        Q, R = jnp.linalg.qr(G.T, mode='reduced')
        G = jnp.reshape(Q.T, (r, n, -1), order='F')

        r_max = jnp.max(jnp.abs(R))
        p = (jnp.floor(jnp.log2(r_max))).astype(int)
        R = R / 2.**p

        return R.T, (G, p)

    Yl, Ym, Yr = Y
    r, n = Yr.shape[:2]

    R, (Yr, pr) = body(jnp.ones((1, 1)), Yr)
    R, (Ym, pm) = jax.lax.scan(body, R, Ym, reverse=True)

    Yl = jnp.reshape(Yl, (n, r), order='F')
    Yl = Yl @ R
    Yl = jnp.reshape(Yl, (1, n, r), order='F')

    r_max = jnp.max(jnp.abs(Yl))
    pl = (jnp.floor(jnp.log2(r_max)))
    Yl = Yl / 2.**pl

    return [Yl, Ym, Yr], jnp.hstack((pl, pm, pr))
