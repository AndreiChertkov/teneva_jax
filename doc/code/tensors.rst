Module tensors: collection of explicit useful TT-tensors
--------------------------------------------------------


.. automodule:: teneva_jax.tensors


-----




|
|

.. autofunction:: teneva_jax.tensors.rand

  **Examples**:

  .. code-block:: python

    d = 6                            # Dimension of the tensor
    n = 5                            # Shape of the tensor
    r = 4                            # TT-rank for the TT-tensor
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)    # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # 

  We may use custom limits:

  .. code-block:: python

    d = 6                            # Dimension of the tensor
    n = 5                            # Shape of the tensor
    r = 4                            # TT-rank for the TT-tensor
    a = 0.99                         # Minimum value
    b = 1.                           # Maximum value
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key, a, b)
    print(Y[0])                      # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # [[[0.99905933 0.99505376 0.99201173 0.99603783]
    #   [0.9982403  0.99355506 0.9977989  0.99978416]
    #   [0.99381576 0.99769924 0.99593848 0.99955382]
    #   [0.99640582 0.99803304 0.99341177 0.99905888]
    #   [0.99696002 0.99767435 0.99508183 0.99683427]]]
    # 




|
|

.. autofunction:: teneva_jax.tensors.rand_norm

  **Examples**:

  .. code-block:: python

    d = 6                               # Dimension of the tensor
    n = 5                               # Shape of the tensor
    r = 4                               # TT-rank for the TT-tensor
    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d, n, r, key)  # Build the random TT-tensor
    teneva.show(Y)                      # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # 

  We may use custom limits:

  .. code-block:: python

    d = 6                               # Dimension of the tensor
    n = 5                               # Shape of the tensor
    r = 4                               # TT-rank for the TT-tensor
    m = 42.                             # Mean ("centre")
    s = 0.0001                          # Standard deviation
    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d, n, r, key, m, s)
    print(Y[0])                         # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # [[[42.00022745 42.00018383 41.99995424 41.99999947]
    #   [42.00010626 42.00004057 42.00015906 41.99983497]
    #   [42.00001789 41.99989299 42.00008431 41.99996506]
    #   [42.00011325 41.99989364 41.9999467  42.00013334]
    #   [41.99989569 42.0000333  42.00003193 42.00000196]]]
    # 




|
|

