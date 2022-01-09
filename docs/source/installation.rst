######################
Installation Guide
######################

You can easily install ``CytofDR`` with just one command! This allows you to perform many DR methods as
a one-stop solution. Follow the guide here to get started!

---------

***********
Conda
***********

We plan on releasing this package on ``conda``. Stay tuned!


---------

***********
PyPI
***********

We plan on releasing this package on ``PyPI``. Stay tuned!

---------

*********************
Core Dependencies
*********************

As an omnibus package, we naturally require lots of dependencies. However, due to inconsistencies
with packaging, we only require some core dependencies to be installed. This is the easiest for
users. We also list some additional dependencies that need manual installation and care to get
working! We will walk you through both processes!

The core dependencies are required for ``CytofDR``. They should be automatically installed with
``pip`` or ``conda`` processes list above, but if there is an issue, you can elect to install on
your own.

* scikit-learn
* numpy
* scipy
* umap-learn
* openTSNE
* phate

-------------

***********************
Optional Dependencies
***********************

There are some optional dependencies for additional DR methods that we can support.
They will not affect other core methods. If you want to use them and integrate them into this
package, follow each of the guides below individually, and be sure to check the links to the
original repositories and guides for intsllation.


ZIFA
------

Zero-Inflated Factor Analysis (ZIFA) can be easily installed from `this repository <>`_ by the original
authors (Pierson & Yau, 2015). This package is compatible with our core dependencies. To install,

.. code-block:: shell 

    git clone https://github.com/epierson9/ZIFA
    cd ZIFA
    python setup.py install


SAUCIE
-------

Although SAUCIE performs quite well, it does not have compatibility with our core dependencies. Some care
is needed to install from source. The `repository <https://github.com/KrishnaswamyLab/SAUCIE>`_ is not
currently packaged (nor does it have a proper open-source license for us to do be able to do anything).
Please follow the instructions from the original authors.

.. note::

    Only tensoflow 1.x is supported. This may cause issues with other dependencies and python version of
    ``CytofDR``.

.. warning::

    ``SAUCIE`` does not have an open-source license. Use at your own risk.


GrandPrix
----------

``GrandPrix`` can be installed from source with the original authors' `GitHub repository <https://github.com/ManchesterBioinference/GrandPrix>`_
(Ahmed et al., 2019). Dependency compatibility with ``CytofDR`` is currently untested. We recommend using tensorflow 1.x in a new environment,
just like the case with ``SAUCIE``. To install, you can simply use the following:

.. code-block::

    git clone https://github.com/ManchesterBioinference/GrandPrix
    cd GrandPrix
    pip install tensorflow==1.14
    pip install gpflow
    python setup.py install

.. note::

    The original authors recommend installing GPflow from source. We recommend installing from a ``pip`` or ``conda``
    for easier installation.


FIt-SNE
--------

This is the original implementation of Fourier-Interpolated t-SNE (Linderman et al., 2019), which can be found
[here](https://github.com/KlugerLab/FIt-SNE). Unless you would like to sequeeze the last bit of performance out
of FIt-SNE, we recommend the python package ``openTSNE``, which already ships with ``CytofDR``.

FIt-SNE is technically compatible with ``CytofDR``. However, it is not a Python package. Rather, it's a Python
wrapper. To install FIt-SNE: 

1. Clone the FIt-SNE repository.
2. Name the cloned directory as **fitsne**.
3. Compile the program according to instructions provided by the original author.
4. Install ``FFTW`` as a required dependency.
5. Add the folder to Python path or place it where you run Python in your own project directory.

.. note:: 

    For Linux users without root access, see `this issue <https://github.com/KlugerLab/FIt-SNE/issues/35>`_.

.. note::

    This compatibility may be deprecated in the future.


BH t-SNE
---------

This refers to the original implementation by van der Maaten (2014), which is linked 
`here <https://github.com/lvdmaaten/bhtsne>`_. Again, we don't recommend this implementation because scikit-learn
already has an implementation and openTSNE is much faster. If you want to use this anyways, do the following:

1. Clone the repository linked above.
2. Compile the C++ files as intructed by the original authors.


scvis
-------

``scvis`` can be installed from the original authors' `GitHub repository <https://github.com/shahcompbio/scvis>`_
(Ding et al., 2018). Like SAUCIE, it is incompatible with ``CytofDR`` because of dependency issues, especially
``tensorflow``. However, this package has **a few serious caveats**:

* ``scvis`` does not work with ``CytofDR`` at all because it only supports its own CLI.
* The installation process has a dependency bug: It only works with tensorflow 1.x, but the automatic installation
  will install the newest tensorflow. You will need to manually reinstall an older version of tensorflow or
  modify ``setup.py``.

To install, you can do the following:

.. code-block:: shell

    git clone https://github.com/shahcompbio/scvis
    cd scvis
    python setup.py install # Or modify setup.py
    pip install tensorflow==1.14 # Or use conda

.. warning::

    ``scvis`` has serious compatibility issues. Please read the caveats above!

.. warning::

    ``scvis`` does not have an open-source license. Use at your own risk.


