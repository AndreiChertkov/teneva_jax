Description of functions and examples
=====================================

Below, we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the TT-cores "[Yl, Ym, Yr]" - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.). Please note that in order to speed up (by orders of magnitude) code compilation (i.e., "jax.jit"), we only support tensors of constant mode size ("n") and TT-rank ("r"). In this case, the tensor ("d > 2") is represented as a list of three jax arrays: "Yl" the first TT-core (3D array of the shape "1xnxr"), an array of all internal TT-cores "Ym" (4D array of the shape "(d-2)xrxnxr"), and the last core "Yr" (3D array of the shape "rxnx1").

Please, also note that all demos assume the following imports (to run them, you should first execute "pip install teneva==0.14.0"; we use the basic teneva package here only to compare the results):

  .. code-block:: python

    from jax.config import config
    config.update('jax_enable_x64', True)

    import os
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'

    import jax
    import jax.numpy as jnp
    import teneva as teneva_base
    import teneva_jax as teneva
    from time import perf_counter as tpc
    rng = jax.random.PRNGKey(42)

-----

.. toctree::
  :maxdepth: 4

  act_one
  act_two
  als
  cross
  data
  maxvol
  sample
  svd
  tensors
  transformation
  vis
