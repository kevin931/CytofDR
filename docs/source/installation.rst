######################
Installation Guide
######################

You can easily install ``CytofDR`` with just one command! This allows you to perform many DR methods as
a one-stop solution. Follow the guide here to get started!

---------

***********
Conda
***********

We are officially on ``conda``!! This is actually our recommended way of installing and running
``CytofDR``. To install, simply run the following:

.. code-block:: shell

    conda install -c kevin931 cytofdr -c conda-forge -c bioconda

If you need to learn more about how to create and manage conda environments, you can take a look
at their `documentation <https://docs.anaconda.com/anaconda/install/>`_.

-----------------

***********
PyPI
***********

Our package is also on ``PyPI``, which you can easily install with the following command:

.. code-block:: shell

    pip install CytofDR

And voila, that's it!

---------

*********************
Core Dependencies
*********************

As an omnibus package, we naturally require lots of dependencies. However, due to inconsistencies
with packaging, we only require some core dependencies to be installed. This is the easiest for
users. We also list some additional dependencies that need manual installation and care to get
working! We will walk you through both processes!

The core dependencies are required for ``CytofDR``. They should be automatically installed with
``pip`` or ``conda`` processes list above, but if there is an issue, you can elect to install them
on your own.

* scikit-learn
* numpy
* scipy
* umap-learn
* openTSNE
* phate
* annoy
* matplotlib
* seaborn

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

Zero-Inflated Factor Analysis (ZIFA) can be easily installed from `this repository <https://github.com/epierson9/ZIFA>`_
by the original authors (Pierson & Yau, 2015). This package is compatible with our core dependencies. To install,

.. code-block:: shell 

    git clone https://github.com/epierson9/ZIFA
    cd ZIFA
    python setup.py install

Then, you will be able to use ZIFA with CytofDR.


SAUCIE
-------

Although SAUCIE performs quite well, it does not have compatibility with our core dependencies. Some care
is needed to install from source. The `repository <https://github.com/KrishnaswamyLab/SAUCIE>`_ is not
currently packaged (nor does it have a proper open-source license for us to do be able to do anything).
To install, first, you will need to have ``python 3.7`` and the core dependencies along with ``CytofDR``
installed in your environment. For this, we **highly recommend** using ``conda`` to manage this enviroment.
Then, you will need to install the following:

.. code-block:: shell

    conda activate your_environment
    conda install tensorflow=1.15 scikit-learn
    conda install -c bioconda fcsparser
    pip install fcswrite

Then, since ``SAUCIE`` is not actually installable, you will need to place it in your working directory
to make it run:

.. code-block:: shell

    git clone https://github.com/KrishnaswamyLab/SAUCIE

These steps should allow you to use ``SAUCIE`` as intended. Of course, you can use ``pip`` if you
prefer.

.. note::

    Only tensoflow 1.x is supported. This may cause issues with other dependencies and python version of
    ``CytofDR`` in the future.

.. warning::

    ``SAUCIE`` has a known issue of being able to run only once after import using ``CytofDR``. We don't
    yet have a workaround for this. Please track this issue `here <https://github.com/kevin931/CytofDR/issues/5>`_.


GrandPrix
----------

``GrandPrix`` can be installed from source with the original authors' `GitHub repository <https://github.com/ManchesterBioinference/GrandPrix>`_
(Ahmed et al., 2019). Again, you will need ``python 3.7`` and ``tensorflow 1.x`` to get this working. To install, you can simply use the following:

.. code-block::

    conda activate your_environment
    conda install tensorflow=1.15
    conda install -c conda-forge gpflow

    git clone https://github.com/ManchesterBioinference/GrandPrix
    cd GrandPrix
    python setup.py install

This should be compatible with ``SAUCIE`` in the same environment.

.. note::

    The original authors recommend installing GPflow from source. We recommend installing from a ``pip`` or ``conda``
    for easier installation.