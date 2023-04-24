Module maxvol: compute the maximal-volume submatrix
---------------------------------------------------


.. automodule:: teneva_jax.maxvol


-----




|
|

.. autofunction:: teneva_jax.maxvol.maxvol

  **Examples**:

  .. code-block:: python

    n = 5000                           # Number of rows
    r = 50                             # Number of columns
    rng, key = jax.random.split(rng)
    A = jax.random.normal(key, (n, r)) # Random tall matrix

  .. code-block:: python

    e = 1.01  # Accuracy parameter
    k = 500   # Maximum number of iterations

  .. code-block:: python

    # Compute row numbers and coefficient matrix:
    I, B = teneva.maxvol(A, e, k)
    
    # Maximal-volume square submatrix:
    C = A[I, :]

  .. code-block:: python

    print(f'|Det C|        : {jnp.abs(jnp.linalg.det(C)):-10.2e}')
    print(f'Max |B|        : {jnp.max(jnp.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {jnp.max(jnp.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', jnp.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # |Det C|        :   1.29e+40
    # Max |B|        :   1.00e+00
    # Max |A - B C|  :   9.10e-15
    # Selected rows  :         50 >  [ 120  315  571  798 1037 1049 1098 1250 1286 1304 1309 1419 1444 1604
    #  1610 1766 1835 1887 1956 2085 2324 2327 2458 2602 2817 2926 3119 3242
    #  3322 3497 3508 3705 3715 3722 3743 3771 3811 3904 3973 4068 4101 4165
    #  4310 4321 4399 4439 4544 4771 4871 4938]
    # 




|
|

.. autofunction:: teneva_jax.maxvol.maxvol_rect

  **Examples**:

  .. code-block:: python

    n = 5000                           # Number of rows
    r = 50                             # Number of columns
    rng, key = jax.random.split(rng)
    A = jax.random.normal(key, (n, r)) # Random tall matrix

  .. code-block:: python

    e = 1.01    # Accuracy parameter
    dr_min = 2  # Minimum number of added rows
    dr_max = 8  # Maximum number of added rows
    e0 = 1.05   # Accuracy parameter for the original maxvol algorithm
    k0 = 50     # Maximum number of iterations for the original maxvol algorithm

  THIS IS DRAFT !!!

  .. code-block:: python

    # Row numbers and coefficient matrix:
    I, B = teneva.maxvol_rect(A, e,
        dr_min, dr_max, e0, k0)
    
    # Maximal-volume rectangular submatrix:
    C = A[I, :]

    # >>> ----------------------------------------
    # >>> Output:

    # /Users/andrei/opt/anaconda3/envs/teneva_jax/lib/python3.8/site-packages/jax-0.4.8-py3.8.egg/jax/_src/ops/scatter.py:89: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=float64 to dtype=int64. In future JAX releases this will result in an error.
    #   warnings.warn("scatter inputs have incompatible types: cannot safely cast "
    # 




|
|

