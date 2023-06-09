Module vis: visualization methods for tensors
---------------------------------------------


.. automodule:: teneva_jax.vis


-----




|
|

.. autofunction:: teneva_jax.vis.show

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with mode size 4 and TT-rank 3:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(5, 4, 3, key)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     4 | r =     3 |
    # 

  If an incorrect TT-tensor is passed to the function (the correctness of the shape of all cores is explicitly checked), then an error will be generated:

  .. code-block:: python

    Y = []
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid TT-tensor
    # 

  .. code-block:: python

    Y = [42.]
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid TT-tensor
    # 

  .. code-block:: python

    Y = [
        jnp.zeros((1, 5, 7)),
        jnp.zeros((100, 42, 7, 1)),
        jnp.zeros((42, 7, 1))]
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid shape of middle cores for TT-tensor
    # 

  .. code-block:: python

    import numpy as onp # Numpy is not supported!
    
    Y = [
        onp.zeros((1, 5, 3)),
        onp.zeros((100, 3, 5, 3)),
        onp.zeros((3, 5, 1))]
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid left core of TT-tensor
    # 

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(5, 6, 7, key)
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : TT-rank should be no greater than mode size
    # 




|
|

