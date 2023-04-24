# workflow

> Workflow instructions for `teneva_jax` developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name teneva_jax python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva_jax
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab matplotlib teneva==0.14.0
    ```

5. Install teneva_jax:
    ```bash
    python setup.py install
    ```

6. Reinstall teneva_jax (after updates of the code):
    ```bash
    clear && pip uninstall teneva_jax -y && python setup.py install
    ```

7. Rebuild the docs (after updates of the code):
    ```bash
    python doc/build.py
    ```

8. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva_jax --all -y
    ```


## How to add a new function

1. Choose the most suitable module from `teneva_jax` folder

2. Choose the name for function in lowercase

3. Add new function in alphabetical order, separating it with two empty lines from neighboring functions

4. Add function in alphabetical order into `__init__.py`

5. Make documentation (i.e., `docstring`) for the function similar to other functions

6. Prepare a demo for a function (jupyter notebook in the `demo` folder) similar to demos for other functions in the jupyter notebook with the same name as a module name (add it in alphabetical order)
    > Note that it's important to use a consistent style for all functions, as the code is then automatically exported from the jupyter notebooks to assemble the online documentation.

7. Add function name into dict in docs `doc/map.py` and rebuild the docs (run `python doc/build.py`), check the result in web browser (see `doc/_build/html/index.html`)

8. Make commit

9. Use the new function locally until update of the package version


## How to update the package version

1. Build the docs `python doc/build.py`

2. Update version (like `0.1.X`) in the file `teneva_jax/__init__.py`

    > For breaking changes we should increase the major index (`1`), for non-breaking changes we should increase the minor index (`X`)

3. Build the docs `python doc/build.py`

4. Do commit `Update version (0.1.X)` and push

5. Upload new version to `pypi` (login: AndreiChertkov)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

6. Reinstall and check that installed version is new
    ```bash
    pip install --no-cache-dir --upgrade teneva_jax
    ```

7. Check the [teneva_jax docs build](https://readthedocs.org/projects/teneva_jax/builds/)

8. Check the [teneva_jax docs site](https://teneva-jax.readthedocs.io/)
