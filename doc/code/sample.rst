Module sample: random sampling for/from the TT-tensor
-----------------------------------------------------


.. automodule:: teneva_jax.sample


-----




|
|

.. autofunction:: teneva_jax.sample.sample

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zl, zm = teneva.interface_rtl(Y)
    
    rng, key = jax.random.split(rng)
    i = teneva.sample(Y, zm, key)
    print(i)

    # >>> ----------------------------------------
    # >>> Output:

    # [0 4 1 4 0 2 4 1]
    # 

  And now let check this function for big random TT-tensor:

  .. code-block:: python

    interface_rtl = jax.jit(teneva.interface_rtl)
    sample = jax.jit(jax.vmap(teneva.sample, (None, None, 0)))

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)

  .. code-block:: python

    zl, zm = interface_rtl(Y)
    
    m = 10  # Number of samples
    rng, key = jax.random.split(rng)
    I = sample(Y, zm, jax.random.split(key, m))
    
    for i in I: # i is a sample of the length d = 1000
        print(len(i), jnp.mean(i))

    # >>> ----------------------------------------
    # >>> Output:

    # 1000 48.005
    # 1000 48.943
    # 1000 50.079
    # 1000 50.75
    # 1000 48.632
    # 1000 49.833
    # 1000 50.394
    # 1000 49.366
    # 1000 49.688
    # 1000 49.441
    # 

  Let compare this function with numpy realization:

  .. code-block:: python

    d = 25       # Dimension of the tensor
    n = 10       # Mode size of the tensor
    r = 5        # Rank of the tensor
    m = 100000   # Number of samples

  .. code-block:: python

    Y_base = teneva_base.rand([n]*d, r)

  .. code-block:: python

    t = tpc()
    I_base = teneva_base.sample(Y_base, m)
    t = tpc() - t
    
    print(f'Time : {t:-8.2f}')
    print(f'Mean : {jnp.mean(I_base):-8.2f}')
    print(f'Var  : {jnp.var(I_base):-8.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Time :    54.39
    # Mean :     4.65
    # Var  :     7.59
    # 

  .. code-block:: python

    Y = teneva.convert(Y_base) # Convert it to the jax version

  .. code-block:: python

    t = tpc()
    interface_rtl = jax.jit(teneva.interface_rtl)
    sample = jax.jit(jax.vmap(teneva.sample, (None, None, 0)))
    
    zl, zm = interface_rtl(Y)
    rng, key = jax.random.split(rng)
    I = sample(Y, zm, jax.random.split(key, m))
    t = tpc() - t
    
    print(f'Time : {t:-8.2f}')
    print(f'Mean : {jnp.mean(I):-8.2f}')
    print(f'Var  : {jnp.var(I):-8.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Time :     1.64
    # Mean :     4.65
    # Var  :     7.66
    # 




|
|

.. autofunction:: teneva_jax.sample.sample_lhs

  **Examples**:

  .. code-block:: python

    d = 3  # Dimension of the tensor/grid
    n = 5  # Shape of the tensor/grid
    m = 8  # Number of samples
    
    rng, key = jax.random.split(rng)
    I = teneva.sample_lhs(d, n, m, key)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[4 1 1]
    #  [3 0 0]
    #  [2 1 0]
    #  [0 3 4]
    #  [1 2 2]
    #  [3 2 3]
    #  [4 4 3]
    #  [0 3 1]]
    # 




|
|

.. autofunction:: teneva_jax.sample.sample_rand

  **Examples**:

  .. code-block:: python

    d = 3  # Dimension of the tensor/grid
    n = 5  # Shape of the tensor/grid
    m = 8  # Number of samples
    
    rng, key = jax.random.split(rng)
    I = teneva.sample_rand(d, n, m, key)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[3 1 2]
    #  [3 1 1]
    #  [3 1 1]
    #  [1 2 2]
    #  [4 3 3]
    #  [4 4 1]
    #  [3 0 1]
    #  [2 4 4]]
    # 




|
|

