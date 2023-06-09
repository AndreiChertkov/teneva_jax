"""Package teneva, module data: functions for working with datasets.

This module contains functions for working with datasets, including
"accuracy_on_data" function.

"""
import jax
import jax.numpy as jnp
import teneva_jax as teneva


def accuracy_on_data(Y, I_data, y_data):
    """Compute the relative error of TT-tensor on the dataset.

    Args:
        I_data (jnp.ndarray): multi-indices for items of dataset in the form of
            array of the shape [samples, d].
        y_data (jnp.ndarray): values for items related to I_data of dataset in
            the form of array of the shape [samples].

    Returns:
        jnp.ndarray of size 1: the relative error.

    Note:
        If I_data or y_data is not provided, the function will return -1.

    """
    y = teneva.get_many(Y, I_data)
    return jnp.linalg.norm(y - y_data) / jnp.linalg.norm(y_data)
