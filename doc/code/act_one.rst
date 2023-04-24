Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva_jax.act_one


-----




|
|

.. autofunction:: teneva_jax.act_one.convert

  **Examples**:

  .. code-block:: python

    import numpy as onp

  Let build jax TT-tensor and convert it to numpy (base) version:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(6, 5, 4, key)
    teneva.show(Y)
    
    print('Is jax   : ', isinstance(Y[0], jnp.ndarray))
    print('Is numpy : ', isinstance(Y[0], onp.ndarray))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Is jax   :  True
    # Is numpy :  False
    # 

  .. code-block:: python

    Y_base = teneva.convert(Y)
    teneva_base.show(Y_base)
    
    print('Is jax   : ', isinstance(Y_base[0], jnp.ndarray))
    print('Is numpy : ', isinstance(Y_base[0], onp.ndarray))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     6D : |5| |5| |5| |5| |5| |5|
    # <rank>  =    4.0 :   \4/ \4/ \4/ \4/ \4/
    # Is jax   :  False
    # Is numpy :  True
    # 

  And now let convert the numpy (base) TT-tensor back into jax format:

  .. code-block:: python

    Z = teneva.convert(Y_base)
    teneva.show(Z)
    
    # Check that it is the same:
    e = jnp.max(jnp.abs(teneva.full(Y) - teneva.full(Z)))
    
    print('Is jax   : ', isinstance(Z[0], jnp.ndarray))
    print('Is numpy : ', isinstance(Z[0], onp.ndarray))
    print('Error    : ', e)   

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Is jax   :  True
    # Is numpy :  False
    # Error    :  0.0
    # 




|
|

.. autofunction:: teneva_jax.act_one.copy

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with mode size 4 and TT-rank 12:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(10, 9, 7, key)
    
    Z = teneva.copy(Y) # The copy of Y  
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.37622900483374
    # -0.37622900483374
    # 




|
|

.. autofunction:: teneva_jax.act_one.get

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = jnp.arange(2**d) # Tensor will be 2^d
    Y0 = jnp.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    k = jnp.array([0, 1, 0, 1, 0])
    y1 = teneva.get(Y1, k)
    
    # Compute the same element of the original tensor:
    y0 = Y0[tuple(k)]
    
    # Compare values:
    e = jnp.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # Error : 6.7e-16
    # 

  Let compare this function with numpy realization:

  .. code-block:: python

    Y1_base = teneva.convert(Y1) # Convert tensor to numpy version
    y1_base = teneva_base.get(Y1_base, k)
    
    print(y1)
    print(y1_base)

    # >>> ----------------------------------------
    # >>> Output:

    # -0.8390715290764531
    # -0.8390715290764531
    # 




|
|

.. autofunction:: teneva_jax.act_one.get_log

  **Examples**:

  .. code-block:: python

    d = 6  # Dimension of the tensor
    n = 5  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct random d-dim non-negative TT-tensor:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    Y = teneva.mul(Y, Y)
    
    # Print the TT-tensor:
    teneva.show(Y)
    
    # Compute the full tensor from the TT-tensor:  
    Y0 = teneva.full(Y)
    
    # Select some tensor element and compute the value:
    k = jnp.array([3, 1, 2, 1, 0, 4])
    y1 = teneva.get_log(Y, k)
    
    # Compute the same element of the original tensor:
    y0 = jnp.log(Y0[tuple(k)])
    
    # Compare values:
    e = jnp.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Error : 0.0e+00
    # 

  We may also use vmap and jit for this function:

  .. code-block:: python

    d = 10   # Dimension of the tensor
    n = 10   # Mode size of the tensor
    r = 3    # Rank of the tensor
    m = 1000 # Batch size
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    Y = teneva.mul(Y, Y)
    
    rng, key = jax.random.split(rng)
    K = teneva.sample_lhs(d, n, m, key)
    
    get_log = jax.vmap(jax.jit(teneva.get_log), (None, 0))
    y = get_log(Y, K)
    print(y[:2])

    # >>> ----------------------------------------
    # >>> Output:

    # [-2.64939165 -0.71116583]
    # 




|
|

.. autofunction:: teneva_jax.act_one.get_many

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = jnp.arange(2**d) # Tensor will be 2^d
    Y0 = jnp.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    K = jnp.array([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    y1 = teneva.get_many(Y1, K)
    
    # Compute the same elements of the original tensor:
    y0 = jnp.array([Y0[tuple(k)] for k in K])
    
    # Compare values:
    e = jnp.max(jnp.abs(y1-y0))
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # Error : 8.9e-16
    # 

  We can compare the calculation time using the base function ("get") with "jax.vmap" and the function "get_many":

  .. code-block:: python

    d = 1000   # Dimension of the tensor
    n = 100    # Mode size of the tensor
    r = 10     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    get1 = jax.jit(jax.vmap(teneva.get, (None, 0)))
    get2 = jax.jit(teneva.get_many)
    
    for m in [1, 1.E+1, 1.E+2, 1.E+3, 1.E+4]:
        # TODO: remove teneva_base here
        I = jnp.array(teneva_base.sample_lhs([n]*d, int(m)))
    
        t1 = tpc()
        y1 = get1(Y, I)
        t1 = tpc() - t1
    
        t2 = tpc()
        y2 = get2(Y, I)
        t2 = tpc() - t2
    
        print(f'm: {m:-7.1e} | T1 : {t1:-8.4f} | T2 : {t2:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    # m: 1.0e+00 | T1 :   0.0602 | T2 :   0.0593
    # m: 1.0e+01 | T1 :   0.0858 | T2 :   0.0863
    # m: 1.0e+02 | T1 :   0.1060 | T2 :   0.1050
    # m: 1.0e+03 | T1 :   0.1555 | T2 :   0.1624
    # m: 1.0e+04 | T1 :   0.7481 | T2 :   0.7820
    # 




|
|

.. autofunction:: teneva_jax.act_one.get_stab

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = jnp.arange(2**d) # Tensor will be 2^d
    Y0 = jnp.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    k = jnp.array([0, 1, 0, 1, 0])
    y1, p1 = teneva.get_stab(Y1, k)
    print(y1)
    print(p1)
    
    # Reconstruct the value:
    y1 = y1 * 2.**jnp.sum(p1)
    print(y1)
    
    # Compute the same element of the original tensor:
    y0 = Y0[tuple(k)]
    
    # Compare values:
    e = jnp.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # -1.6781430581529062
    # [ 0.  0.  0. -1.  0.]
    # -0.8390715290764531
    # Error : 6.7e-16
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    k = jnp.zeros(1000, dtype=jnp.int32)
    y, p = teneva.get_stab(Y, k)
    print(y, jnp.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.1007570174016164 799.0
    # 




|
|

.. autofunction:: teneva_jax.act_one.grad

  **Examples**:

  .. code-block:: python

    l = 1.E-4   # Learning rate
    d = 5       # Dimension of the tensor
    n = 4       # Mode size of the tensor
    r = 2       # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key=key)
    
    # Targer multi-index for gradient:
    i = jnp.array([0, 1, 2, 3, 0])
    
    y = teneva.get(Y, i)
    dY = teneva.grad(Y, i)

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    y_base, dY_base = teneva_base.get_and_grad(Y_base, i)
    dY_base = [G[:, k, :] for G, k in zip(dY_base, i)]
    dY_base = [dY_base[0], jnp.array(dY_base[1:-1]), dY_base[-1]]
    print('Error : ', jnp.max(jnp.array([jnp.max(jnp.abs(g-g_base)) for g, g_base in zip(dY, dY_base)])))

    # >>> ----------------------------------------
    # >>> Output:

    # Error :  6.938893903907228e-18
    # 

  Let apply the gradient:

  .. code-block:: python

    Z = teneva.copy(Y) # TODO
    Z[0] = Z[0].at[:, i[0], :].add(-l * dY[0])
    for k in range(1, d-1):
        Z[1] = Z[1].at[k-1, :, i[k], :].add(-l * dY[1][k-1])
    Z[2] = Z[2].at[:, i[d-1], :].add(-l * dY[2])
    
    z = teneva.get(Z, i)
    e = jnp.max(jnp.abs(teneva.full(Y) - teneva.full(Z)))
    
    print(f'Old value at multi-index : {y:-12.5e}')
    print(f'New value at multi-index : {z:-12.5e}')
    print(f'Difference for tensors   : {e:-12.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Old value at multi-index : -1.22741e-02
    # New value at multi-index : -1.22785e-02
    # Difference for tensors   :      2.6e-05
    # 




|
|

.. autofunction:: teneva_jax.act_one.interface_ltr

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zm, zr = teneva.interface_ltr(Y)
    
    for z in zm:
        print(z)
    print(zr)

    # >>> ----------------------------------------
    # >>> Output:

    # [-0.16194455  0.01090809  0.97433024  0.1559986 ]
    # [ 0.67701845 -0.17267149  0.46267419  0.5456768 ]
    # [-0.6388783  -0.61156506 -0.43463404  0.1700469 ]
    # [-0.58225588  0.11947942  0.05768481 -0.80210674]
    # [-0.32840251  0.8625416   0.19736311  0.3304869 ]
    # [ 0.53878607 -0.53108736  0.27270038  0.59438228]
    # [ 0.8023033  -0.42153121 -0.33042209 -0.26351867]
    # 

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    phi_l = teneva_base.interface(Y_base, ltr=True)
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # [1.]
    # [-0.16194455  0.01090809  0.97433024  0.1559986 ]
    # [ 0.67701845 -0.17267149  0.46267419  0.5456768 ]
    # [-0.6388783  -0.61156506 -0.43463404  0.1700469 ]
    # [-0.58225588  0.11947942  0.05768481 -0.80210674]
    # [-0.32840251  0.8625416   0.19736311  0.3304869 ]
    # [ 0.53878607 -0.53108736  0.27270038  0.59438228]
    # [ 0.8023033  -0.42153121 -0.33042209 -0.26351867]
    # [-1.]
    # 




|
|

.. autofunction:: teneva_jax.act_one.interface_rtl

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zl, zm = teneva.interface_rtl(Y)
    
    print(zl)
    for z in zm:
        print(z)

    # >>> ----------------------------------------
    # >>> Output:

    # [-0.88230513 -0.25794634 -0.22432639  0.32354136]
    # [-0.55980014 -0.15127749  0.74733168  0.32439836]
    # [-0.19815258 -0.19100523  0.73302586  0.62203348]
    # [ 0.58566709  0.38105157 -0.21170587 -0.68335524]
    # [ 0.26252681  0.77866465 -0.02860785 -0.56915958]
    # [-0.04265967 -0.49325475  0.24061258  0.83485657]
    # [-0.4873671  -0.4158353   0.35158957  0.68259731]
    # 

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    phi_r = teneva_base.interface(Y_base, ltr=False)
    for phi in phi_r:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # [-1.]
    # [-0.88230513 -0.25794634 -0.22432639  0.32354136]
    # [-0.55980014 -0.15127749  0.74733168  0.32439836]
    # [-0.19815258 -0.19100523  0.73302586  0.62203348]
    # [ 0.58566709  0.38105157 -0.21170587 -0.68335524]
    # [ 0.26252681  0.77866465 -0.02860785 -0.56915958]
    # [-0.04265967 -0.49325475  0.24061258  0.83485657]
    # [-0.4873671  -0.4158353   0.35158957  0.68259731]
    # [1.]
    # 




|
|

.. autofunction:: teneva_jax.act_one.mean

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m = teneva.mean(Y)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = jnp.mean(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 1.41e-18
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    teneva.mean(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # Array(0., dtype=float64)
    # 




|
|

.. autofunction:: teneva_jax.act_one.mean_stab

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m, p = teneva.mean_stab(Y)
    print(m)
    print(p)
    
    # Reconstruct the value:
    m = m * 2.**jnp.sum(p)
    print(m)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = jnp.mean(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # -1.3138548907194174
    # [-2. -1. -1.  0. -2. -1.]
    # -0.010264491333745449
    # Error     : 5.20e-18
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    m, p = teneva.mean_stab(Y)
    print(m, jnp.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.2086063088986352 -2530.0
    # 




|
|

.. autofunction:: teneva_jax.act_one.norm

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r = 3   # TT-rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)

  .. code-block:: python

    v = teneva.norm(Y)  # Compute the Frobenius norm
    print(v)            # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [50.28527148]
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)
    
    v_full = jnp.linalg.norm(Y_full)
    print(v_full)
    
    e = abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}') 

    # >>> ----------------------------------------
    # >>> Output:

    # 50.28527148425206
    # Error     : 8.48e-16
    # 




|
|

.. autofunction:: teneva_jax.act_one.norm_stab

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r = 3   # Rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)

  .. code-block:: python

    v, p = teneva.norm_stab(Y) # Compute the Frobenius norm
    print(v) # Print the scaled value
    print(p) # Print the scale factors
    
    v = v * 2**jnp.sum(p) # Resulting value
    print(v)   # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [1.03970497]
    # [0.5 1.  1.5 1.  1.5]
    # [47.05167577]
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)
    
    v_full = jnp.linalg.norm(Y_full)
    print(v_full)
    
    e = abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}') 

    # >>> ----------------------------------------
    # >>> Output:

    # 47.05167576892552
    # Error     : 9.06e-16
    # 




|
|

.. autofunction:: teneva_jax.act_one.sum

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m = teneva.sum(Y)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = jnp.sum(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 1.81e-13
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    teneva.sum(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # Array(0., dtype=float64)
    # 




|
|

.. autofunction:: teneva_jax.act_one.sum_stab

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m, p = teneva.sum_stab(Y)
    print(m)
    print(p)
    
    # Reconstruct the value:
    m = m * 2.**jnp.sum(p)
    print(m)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = jnp.sum(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # 1.7145681216308906
    # [0. 1. 1. 1. 2. 0.]
    # 54.8661798921885
    # Error     : 2.56e-13
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    m, p = teneva.sum_stab(Y)
    print(m, jnp.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.4416615020667247 -2538.0
    # 




|
|

