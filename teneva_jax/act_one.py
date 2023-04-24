"""Package teneva, module act_one: single TT-tensor operations.

This module contains the basic operations with one TT-tensor (Y), including
"copy", "get", "sum", etc.

"""
import jax
import jax.numpy as jnp
try:
    import numpy as np
    WITH_NUMPY = True
except Exception as e:
    WITH_NUMPY = False
import teneva_jax as teneva


def convert(Y):
    """Convert TT-tensor from base (numpy) format and back.

    Args:
        Y (list): TT-tensor in numpy format (a list of d ordinary numpy arrays)
            or in jax format (a list of 3 jax.numpy arrays).

    Returns:
        list: TT-tensor in numpy format if Y is in jax format and vice versa.

    """
    if not isinstance(Y[0], jnp.ndarray): # Ordinary numpy format -> jax
        Yl = jnp.array(Y[0], copy=True)
        Ym = jnp.array(Y[1:-1], copy=True)
        Yr = jnp.array(Y[-1], copy=True)
        return [Yl, Ym, Yr]
    else:                                # Jax format -> ordinary numpy format
        if not WITH_NUMPY:
            raise ValueError('Numpy is required for this function')
        Yl, Ym, Yr = Y
        Ym_base = jnp.split(Ym, Ym.shape[0])
        Yl = np.array(Yl)
        for k in range(len(Ym_base)):
            Ym_base[k] = np.array(Ym_base[k][0])
        Yr = np.array(Yr)
        return [Yl] + Ym_base + [Yr]


def copy(Y):
    """Return a copy of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        list: TT-tensor, which is a copy of the given TT-tensor.

    """
    return [Y[0].copy(), Y[1].copy(), Y[2].copy()]


def get(Y, k):
    """Compute the element of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (jnp.ndarray): the multi-index for the tensor of the length d.

    Returns:
        jnp.ndarray of size 1: the element of the TT-tensor.

    """
    def body(q, data):
        i, G = data
        q = jnp.einsum('q,qr->r', q, G[:, i, :])
        return q, None

    Yl, Ym, Yr = Y

    q = Yl[0, k[0], :]
    q, _ = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, _ = body(q, (k[-1], Yr))

    return q[0]


def get_log(Y, k):
    """Compute the logarithm of the element of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (jnp.ndarray): the multi-index for the tensor of the length d.

    Returns:
        jnp.ndarray of size 1: the logarithm of the element of the TT-tensor.

    """
    def body(q, data):
        i, G = data
        Q = jnp.einsum('r,riq->iq', q, G)
        p = jnp.sqrt(jnp.sum(Q**2, axis=1)[i])
        q = (q @ G[:, i, :]) / p
        return q, p

    Yl, Ym, Yr = Y

    q, pl = body(jnp.ones(1), (k[0], Yl))
    q, pm = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, pr = body(q, (k[-1], Yr))

    y = jnp.hstack((pl, pm, pr, jnp.linalg.norm(q)))
    return jnp.sum(jnp.log(y))


def get_many(Y, K):
    """Compute the elements of the TT-tensor on many multi-indices.

    Args:
        Y (list): d-dimensional TT-tensor.
        K (jnp.ndarray): the multi-indices for the tensor in the of the shape
            [samples, d].

    Returns:
        jnp.ndarray: the elements of the TT-tensor for multi-indices K (array
        of the length samples).

    """
    def body(Q, data):
        i, G = data
        Q = jnp.einsum('kq,qkr->kr', Q, G[:, i, :])
        return Q, None

    Yl, Ym, Yr = Y

    Q = Yl[0, K[:, 0], :]
    Q, _ = jax.lax.scan(body, Q, (K[:, 1:-1].T, Ym))
    Q, _ = body(Q, (K[:, -1], Yr))

    return Q[:, 0]


def get_stab(Y, k):
    """Compute the element of the TT-tensor with stabilization factor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (jnp.ndarray): the multi-index for the tensor of the length d.

    Returns:
        (jnp.ndarray of size 1, jnp.ndarray): the scaled value of the TT-tensor
        v and stabilization factor p for each TT-core (array of the length d).
        The resulting value is v * 2^{sum(p)}.

    """
    def body(q, data):
        i, G = data
        q = jnp.einsum('q,qr->r', q, G[:, i, :])

        q_max = jnp.max(jnp.abs(q))
        p = jnp.floor(jnp.log2(q_max))
        q = q / 2.**p

        return q, p

    Yl, Ym, Yr = Y

    q, pl = Yl[0, k[0], :], 0
    q, pm = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, pr = body(q, (k[-1], Yr))

    return q[0], jnp.hstack((pl, pm, pr))


def grad(Y, k):
    """Compute gradients of the TT-tensor for given multi-index.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (list, jnp.ndarray): the multi-index for the tensor.

    Returns:
        list: the matrices which collect the gradients for all TT-cores.

    Todo:
        Move z construction into separate interface_* functions.

    """
    def body_ltr(z, data):
        G, i = data
        z = z @ G[:, i, :]
        return z, z

    Yl, Ym, Yr = Y

    z, zl = body_ltr(jnp.ones(1), (Yl, k[0]))
    z, zm = jax.lax.scan(body_ltr, z, (Ym, k[1:-1]))

    zm_ltr = jnp.vstack((zl, zm[:-1]))
    zr_ltr = zm[-1]

    def body_rtl(z, data):
        G, i = data
        z = G[:, i, :] @ z
        return z, z

    z, zr = body_rtl(jnp.ones(1), (Yr, k[-1]))
    z, zm = jax.lax.scan(body_rtl, z, (Ym, k[1:-1]), reverse=True)

    zl_rtl = zm[0]
    zm_rtl = jnp.vstack((zm[1:], zr))

    def body(z, data):
        zl, zr = data
        Gg = jnp.outer(zl, zr)
        return None, Gg

    _, Gl = body(None, (jnp.ones(1), zl_rtl))
    _, Gm = jax.lax.scan(body, None, (zm_ltr, zm_rtl))
    _, Gr = body(None, (zr_ltr, jnp.ones(1)))

    return [Gl, Gm, Gr]


def interface_ltr(Y):
    """Generate the left to right interface vectors for the TT-tensor Y.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        (list, list): inner interface vectors zl (list of arrrays of the length
        d-2) and the right interface vector zr.

    """
    def body(z, G):
        z = z @ jnp.sum(G, axis=1)
        z /= jnp.linalg.norm(z)
        return z, z

    Yl, Ym = Y[:-1]

    z, zl = body(jnp.ones(1), Yl)
    z, zm = jax.lax.scan(body, z, Ym)

    zr = zm[-1]
    zm = jnp.vstack((zl, zm[:-1]))

    return zm, zr


def interface_rtl(Y):
    """Generate the right to left interface vectors for the TT-tensor Y.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        (list, list): left interface vector zl and inner interface vectors zm
        (list of arrrays of the length d-2).

    """
    def body(z, G):
        z = jnp.sum(G, axis=1) @ z
        z /= jnp.linalg.norm(z)
        return z, z

    Ym, Yr = Y[1:]

    z, zr = body(jnp.ones(1), Yr)
    z, zm = jax.lax.scan(body, z, Ym, reverse=True)

    zl = zm[0]
    zm = jnp.vstack((zm[1:], zr))

    return zl, zm


def mean(Y):
    """Compute mean value of the TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        jnp.ndarray of size 1: the mean value of the TT-tensor.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        q = jnp.ones(k) / k
        R = R @ jnp.einsum('riq,i->rq', Y_cur, q)
        return R, None

    Yl, Ym, Yr = Y
    R, _ = scan(jnp.ones((1, 1)), Yl)
    R, _ = jax.lax.scan(scan, R, Ym)
    R, _ = scan(R, Yr)

    return R[0, 0]


def mean_stab(Y):
    """Compute mean value of the TT-tensor with stabilization factor.

    Args:
        Y (list): TT-tensor with d dimensions.

    Returns:
        (jnp.ndarray of size 1, jnp.ndarray): the scaled mean value of the
        TT-tensor m and stabilization factor p for each TT-core (array of the
        length d). The resulting value is m * 2^{sum(p)}.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        Q = jnp.ones(k) / k
        R = R @ jnp.einsum('riq,i->rq', Y_cur, Q)

        r_max = jnp.max(jnp.abs(R))
        p = jnp.floor(jnp.log2(r_max))
        R = R / 2.**p

        return R, p

    Yl, Ym, Yr = Y
    R, pl = scan(jnp.ones((1, 1)), Yl)
    R, pm = jax.lax.scan(scan, R, Ym)
    R, pr = scan(R, Yr)

    return R[0, 0], jnp.hstack((pl, pm, pr))


def norm(Y, use_stab=False):
    """Compute Frobenius norm of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        jnp.ndarray of size 1: Frobenius norm of the TT-tensor.

    Todo:
        Check negative values from "mul_scalar".

    """
    v = teneva.mul_scalar(Y, Y)
    return jnp.sqrt(v)


def norm_stab(Y):
    """Compute Frobenius norm of the given TT-tensor with stab. factor.

    Args:
        Y (list): TT-tensor.

    Returns:
        (jnp.ndarray of size 1, list): Frobenius norm of the TT-tensor and
        stabilization factor p for each TT-core.

    Todo:
        Check negative values from "mul_scalar".

    """
    v, p = teneva.mul_scalar_stab(Y, Y)
    return jnp.sqrt(v), p/2


def sum(Y):
    """Compute sum of all tensor elements.

    Args:
        Y (list): TT-tensor.

    Returns:
        jnp.ndarray of size 1: the sum of all tensor elements.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        q = jnp.ones(k)
        R = R @ jnp.einsum('riq,i->rq', Y_cur, q)
        return R, None

    Yl, Ym, Yr = Y
    R, _ = scan(jnp.ones((1, 1)), Yl)
    R, _ = jax.lax.scan(scan, R, Ym)
    R, _ = scan(R, Yr)

    return R[0, 0]


def sum_stab(Y):
    """Compute sum of all tensor elements with stabilization factor.

    Args:
        Y (list): TT-tensor with d dimensions.

    Returns:
        (jnp.ndarray of size 1, jnp.ndarray): the scaled sum of all TT-tensor
        elements m and stabilization factor p for each TT-core (array of the
        length d). The resulting value is m * 2^{sum(p)}.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        Q = jnp.ones(k)
        R = R @ jnp.einsum('rmq,m->rq', Y_cur, Q)

        r_max = jnp.max(jnp.abs(R))
        p = (jnp.floor(jnp.log2(r_max)))
        R = R / 2.**p

        return R, p

    Yl, Ym, Yr = Y
    R, pl = scan(jnp.ones((1, 1)), Yl)
    R, pm = jax.lax.scan(scan, R, Ym)
    R, pr = scan(R, Yr)

    return R[0, 0], jnp.hstack((pl, pm, pr))


def _tt_tail_sizes(Y):
    # TMP (will be removed?)
    d = len(Y)
    r = jnp.array([i.shape[0] for i in Y] + [Y[-1].shape[-1]])
    idx_ch = jnp.arange(d)[r[1:] != r[:-1]]
    if len(idx_ch) == 0:
        return 0, 0
    # now len(idx_ch) >= 2

    i_longest = jnp.argmax(idx_ch[1:] - idx_ch[:-1])
    return idx_ch[i_longest] + 1, d - idx_ch[i_longest + 1]
