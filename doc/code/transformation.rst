Module transformation: orthogonalization, truncation and other transformations of the TT-tensors
------------------------------------------------------------------------------------------------


.. automodule:: teneva_jax.transformation


-----




|
|

.. autofunction:: teneva_jax.transformation.full

  **Examples**:

  .. code-block:: python

    d = 5     # Dimension of the tensor
    n = 6     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    teneva.show(Y)
    
    Z = teneva.full(Y)
    
    # Compare one value of original tensor and reconstructed tensor:
    k = jnp.array([0, 1, 2, 3, 4])
    y = teneva.get(Y, k)
    z = Z[tuple(k)]
    e = jnp.abs(z-y)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     6 | r =     4 |
    # Error : 5.6e-17
    # 




|
|

.. autofunction:: teneva_jax.transformation.orthogonalize_rtl

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d=7, n=4, r=3, key=key)
    Z = teneva.orthogonalize_rtl(Y)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     7 | n =     4 | r =     3 |
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    Y_full = teneva.full(Y)
    Z_full = teneva.full(Z)
    e = jnp.max(jnp.abs(Y_full - Z_full))
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.68e-13
    # 

  And we can make sure that all TT-cores, except the first one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    Zl, Zm, Zr = Z
    
    v = [Zl[:, j, :] @ Zl[:, j, :].T for j in range(Zl.shape[1])]
    print(jnp.sum(jnp.array(v), axis=0))
    
    for G in Zm:
        v = [G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]
        print(jnp.sum(jnp.array(v), axis=0))
        
    v = [Zr[:, j, :] @ Zr[:, j, :].T for j in range(Zr.shape[1])]
    print(jnp.sum(jnp.array(v), axis=0))

    # >>> ----------------------------------------
    # >>> Output:

    # [[34549434.73187065]]
    # [[ 1.00000000e+00 -2.08166817e-17  2.77555756e-17]
    #  [-2.08166817e-17  1.00000000e+00  1.38777878e-17]
    #  [ 2.77555756e-17  1.38777878e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -2.77555756e-17 -2.77555756e-17]
    #  [-2.77555756e-17  1.00000000e+00 -1.11022302e-16]
    #  [-2.77555756e-17 -1.11022302e-16  1.00000000e+00]]
    # [[ 1.00000000e+00  2.77555756e-17  4.16333634e-17]
    #  [ 2.77555756e-17  1.00000000e+00 -2.77555756e-17]
    #  [ 4.16333634e-17 -2.77555756e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.66533454e-16 -2.77555756e-17]
    #  [-1.66533454e-16  1.00000000e+00 -2.77555756e-17]
    #  [-2.77555756e-17 -2.77555756e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.80411242e-16  1.11022302e-16]
    #  [-1.80411242e-16  1.00000000e+00 -5.55111512e-17]
    #  [ 1.11022302e-16 -5.55111512e-17  1.00000000e+00]]
    # [[1.00000000e+00 3.12250226e-17 8.32667268e-17]
    #  [3.12250226e-17 1.00000000e+00 2.77555756e-16]
    #  [8.32667268e-17 2.77555756e-16 1.00000000e+00]]
    # 




|
|

.. autofunction:: teneva_jax.transformation.orthogonalize_rtl_stab

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d=7, n=4, r=3, key=key)
    Z_stab, p_stab = teneva.orthogonalize_rtl_stab(Y)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     7 | n =     4 | r =     3 |
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    Z = teneva.copy(Z_stab)
    Z[0] *= 2**jnp.sum(p_stab)
    
    Y_full = teneva.full(Y)
    Z_full = teneva.full(Z)
    e = jnp.max(jnp.abs(Y_full - Z_full))
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 2.56e-13
    # 

  .. code-block:: python

    Zl, Zm, Zr = Z_stab
    
    v = [Zl[:, j, :] @ Zl[:, j, :].T for j in range(Zl.shape[1])]
    print(jnp.sum(jnp.array(v), axis=0))
    
    for G in Zm:
        v = [G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]
        print(jnp.sum(jnp.array(v), axis=0))
        
    v = [Zr[:, j, :] @ Zr[:, j, :].T for j in range(Zr.shape[1])]
    print(jnp.sum(jnp.array(v), axis=0))

    # >>> ----------------------------------------
    # >>> Output:

    # [[7.15816805]]
    # [[ 1.00000000e+00  1.52655666e-16  0.00000000e+00]
    #  [ 1.52655666e-16  1.00000000e+00 -1.38777878e-17]
    #  [ 0.00000000e+00 -1.38777878e-17  1.00000000e+00]]
    # [[ 1.00000000e+00  5.55111512e-17 -2.77555756e-17]
    #  [ 5.55111512e-17  1.00000000e+00 -2.77555756e-17]
    #  [-2.77555756e-17 -2.77555756e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -6.24500451e-17 -2.77555756e-17]
    #  [-6.24500451e-17  1.00000000e+00  1.38777878e-17]
    #  [-2.77555756e-17  1.38777878e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -4.16333634e-17  0.00000000e+00]
    #  [-4.16333634e-17  1.00000000e+00 -9.71445147e-17]
    #  [ 0.00000000e+00 -9.71445147e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -2.77555756e-17 -1.24900090e-16]
    #  [-2.77555756e-17  1.00000000e+00  0.00000000e+00]
    #  [-1.24900090e-16  0.00000000e+00  1.00000000e+00]]
    # [[1.00000000e+00 1.94289029e-16 5.55111512e-17]
    #  [1.94289029e-16 1.00000000e+00 1.38777878e-17]
    #  [5.55111512e-17 1.38777878e-17 1.00000000e+00]]
    # 




|
|

