Module svd: SVD-based algorithms for matrices and tensors
---------------------------------------------------------


.. automodule:: teneva_jax.svd


-----




|
|

.. autofunction:: teneva_jax.svd.matrix_skeleton

  **Examples**:

  .. code-block:: python

    # Shape of the matrix:
    m, n = 100, 30
    
    # Build random matrix, which has rank 3 as a sum of rank-1 matrices:
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 6)
    u = [jax.random.normal(keys[i], (m, )) for i in range(3)]
    v = [jax.random.normal(keys[i], (m, )) for i in range(3, 6)]
    A = jnp.outer(u[0], v[0]) + jnp.outer(u[1], v[1]) + jnp.outer(u[2], v[2])

  .. code-block:: python

    # Compute skeleton decomp.:
    U, V = teneva.matrix_skeleton(A, r=3)
    
    # Approximation error
    e = jnp.linalg.norm(A - U @ V) / jnp.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :', V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 100)
    # Error      : 2.32e-15
    # 

  .. code-block:: python

    # Compute skeleton decomp with small rank:
    U, V = teneva.matrix_skeleton(A, r=2)
    
    # Approximation error:
    e = jnp.linalg.norm(A - U @ V) / jnp.linalg.norm(A)
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :', V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 100)
    # Error      : 4.49e-01
    # 




|
|

.. autofunction:: teneva_jax.svd.svd

  **Examples**:

  .. code-block:: python

    d = 5                # Dimension number
    t = jnp.arange(2**d) # Tensor will be 2^d
    
    # Construct d-dim full array:
    Z_full = jnp.cos(t).reshape([2] * d, order='F')

  .. code-block:: python

    # Construct TT-tensor by TT-SVD:
    Y = teneva.svd(Z_full, r=2)
    
    # Convert it back to numpy to check result:
    Y_full = teneva.full(Y)
    
    # Compute error for TT-tensor vs full tensor:
    e = jnp.linalg.norm(Y_full - Z_full)
    e /= jnp.linalg.norm(Z_full)

  .. code-block:: python

    # Size of the original tensor:
    print(f'Size (np) : {Z_full.size:-8d}')
    
    # Size of the TT-tensor:
    print(f'Size (tt) : {Y[0].size + Y[1].size + Y[2].size:-8d}') # TODO  
    
    # Rel. error for the TT-tensor vs full tensor:
    print(f'Error     : {e:-8.2e}')               

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :       32
    # Size (tt) :       32
    # Error     : 7.78e-16
    # 

  We can also try a lower rank (it will lead to huge error in this case):

  .. code-block:: python

    # Construct TT-tensor by TT-SVD:
    Y = teneva.svd(Z_full, r=1)
    
    # Convert it back to numpy to check result:
    Y_full = teneva.full(Y)
    
    # Compute error for TT-tensor vs full tensor:
    e = jnp.linalg.norm(Y_full - Z_full)
    e /= jnp.linalg.norm(Z_full)
    
    print(f'Size (np) : {Z_full.size:-8d}')
    print(f'Size (tt) : {Y[0].size + Y[1].size + Y[2].size:-8d}') # TODO   
    print(f'Error     : {e:-8.2e}')  

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :       32
    # Size (tt) :       10
    # Error     : 7.13e-01
    # 

  Note that in jax version rank can not be greater than mode size:

  .. code-block:: python

    try:
        Y = teneva.svd(Z_full, r=3)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Rank can not be greater than mode size
    # 




|
|

