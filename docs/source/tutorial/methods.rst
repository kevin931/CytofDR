DR Methods and Usage
=======================

As the package name suggests, ``CytofDR`` is centered around dimension reduction (DR).
We aim to provide a good interface for anyone to perform DR with ease. Although much
of the package acts as an interface to other more popular packages such as ``scikit-learn``
and ``umap-learn``, the convenience offered here is unmatched because we bring to
you the combined power of both general-purpose DR methdods but also those that are
specifically designed for CyTOF and scRNA-seq.

In this tutorial, we will walk you through the methods available, your options for
customization, and how you can best utilize the package beyond the 
`Quickstart Guide <https://cytofdr.readthedocs.io/en/latest/quickstart.html>`_. For
this tutorial, we will use the following expression matrix:

.. code-block:: python

    >>> expression
    array([[1.73462413, 2.44479204, 0.        , ..., 0.22536523, 1.02089248, 0.1500314 ],
           [0.56619612, 1.52259608, 0.        , ..., 0.31847633, 0.        , 0.        ],
           [0.54875404, 0.        , 0.        , ..., 0.17807296, 0.46455456, 3.55193468],
           ...,
           [0.1630427 , 0.32121831, 0.        , ..., 0.61940005, 0.        , 3.50253287],
           [0.30990439, 2.59020988, 0.11689489, ..., 0.94090453, 0.1383413 , 0.        ],
           [0.71138557, 1.72764796, 0.        , ..., 0.        , 0.        , 0.        ]])

with the following dimensions:

.. code-block:: python

    >>> expression.shape 
    (122924, 34)

----------------------------

********************
DR Methods Overview
********************

We currently suppport 16 unique DR methods along with different variations of the
same methods. Below is a comprehensive list of methods:

=========== =================================== =================== ==============
Name          Method                              Alias              Support    
----------- ----------------------------------- ------------------- --------------
UMAP         ``NonLinearMethods.UMAP``            ``UMAP``             Default
tSNE (FFT)   ``NonLinearMethods.open_tsne``       ``open_tsne``        Default
tSNE (BH)    ``NonLinearMethods.sklearn_tsne``    ``sklearn_tsne``     Default
PHATE        ``NonLinearMethods.phate``           ``PHATE``            Default
PCA          ``LinearMethods.PCA``                ``PCA``              Default
ICA          ``LinearMethods.ICA``                ``ICA``              Default
FA           ``LinearMethods.FA``                 ``FA``               Default
Isomap       ``NonLinearMethods.isomap``          ``Isomap``           Default
MDS          ``NonLinearMethods.MDS``             ``MDS``              Default
LLE          ``NonLinearMethods.LLE``             ``LLE``              Default
KPCA (Poly)  ``NonLinearMethods.kernelPCA``       ``kpca_poly``        Default
KPCA (RBF)   ``NonLinearMethods.kernelPCA``       ``kpca_rbf``         Default
NMF          ``LinearMethods.NMF``                ``NMF``              Default
Spectral     ``NonLinearMethods.spectral``        ``Spectral``         Default
SAUCIE       ``NonLinearMethods.SAUCIE``          ``SAUCIE``           Optional
ZIFA         ``LinearMethods.ZIFA``               ``ZIFA``             Optional
GrandPrix    ``NonLinearMethods.grandprix``       ``GrandPrix``        Optional
=========== =================================== =================== ==============

Here, the ``Method`` refers to the location of our wrapper in the ``dr`` module 
of the package. ``Alias`` is the name used if you use the ``dr.run_dr_methods()``
method to run these methods, and likewise, these are the names stored in the
``reductions`` dictionary of the returned object.

For a comprehensive list of references and links, look at the 
`References <https://cytofdr.readthedocs.io/en/latest/references.html>`_ page.

.. note::
    
    Three methods are optional for this package. To use it, you will need to
    manually install them, as shown in the
    `Installation Guide <https://cytofdr.readthedocs.io/en/latest/installation.html>`_.


***********************
Quick DR with Defaults
***********************

If you don't want to fiddle with settings but just want to kickstart your analyses
with a few lines of code, we hear you and we are here for you! That's why we have
an all-in-one dispatcher: ``dr.run_dr_methods()``. This method allows you to run
all your DR methods with just one simple line of code:


.. code-block:: python

    >>> results = dr.run_dr_methods(expression, methods=["UMAP", "PCA"])
    Running PCA
    Runnign UMAP

And of course, you can add as many methods as you like by adding the **alias** presented
above to the list. The returned object ``objects`` is a 
``Reductions`` object, which is fully documented `here <https://cytofdr.readthedocs.io/en/latest/documentation/dr.html>`_
and explained below.

Despite the simplicity, there are still a few things you can customize. We will list them
below.

The Number of Dimensions
-------------------------

By default, all our embeddings are two-dimensional: this is not only the convention
but also easy to visualize. However, if you so desire to use a different number of
dimensions, you can certainly do so:

.. code-block:: python

    >>> results = dr.run_dr_methods(expression, methods="PCA", out_dims = 3)
    >>> results.reductions["PCA"]
    Running PCA
    array([[ 3.95384698e+00, -5.18932314e-03,  1.99425436e+00],
           [ 4.67605078e+00, -1.34965157e+00, -2.68634708e+00],
           [-2.04514713e+00, -1.26489971e+00, -3.89934577e+00],
           ...,
           [-2.66635013e+00, -2.01899595e+00, -3.85585388e+00],
           [ 5.76069021e+00, -1.24300922e+00, -3.77975868e+00],
           [ 3.30609832e+00, -2.16666682e+00, -1.93277340e+00]])

One thing to note is that not all methods support a different dimension. Namely,
``open_tsne`` and ``SAUCIE`` does not support other dimensions. If you would like
to use a different while still running these two methods with 2D output, we
recommend using the custom mode described below so that there is no ambiguity.


Transform: Embedding New Data
---------------------------------

Some methods support training with a subset of the dataset and mapping new data
onto the embedding. This is incredibly useful with large CyTOF samples and with
samples that come in later! Current, two of our methods ``LLE`` and ``Isomap``
support this feature! In the future, there will be support for more methods.
You can simply specify the ``transform`` option. Suppose we want to use the
first observations to train and embedding the entire dataset with ``transform``:

.. code-block:: python

    >>> train = expression[1:1000,:]
    >>> results = dr.run_dr_methods(train, methods="LLE", out_dims = 2, transform=expression)
    >>> results.reductions["LLE"]
    array([[ 0.0016272 , -0.08276973],
           [ 0.05160762,  0.00221715],
           [ 0.01881232, -0.00114671],
           ...,
           [ 0.01875352, -0.00052259],
           [ 0.05155922,  0.00273084],
           [ 0.05156214,  0.00269989]])

And to check the dimensions:

.. code-block:: python

    >>> results.reductions["LLE"].shape
    (122924, 2)

Indeed, it had embedded the entire dataset, and this will be fast! Of course, it may not
be a good idea to train with less than 1% of the observations, but you can decide for yourself
what is a good trade-off.

--------------------------------

****************************************
Working with the ``Reductions`` Object
****************************************

We created the ``Reductions`` object so that you can conveniently manage your embeddings and
their evaluations in one place. As a starter, you may notice that the return type of 
``dr.run_dr_methods()`` is a ``Reductions`` object:

.. code-block:: python

    >>> results = dr.LinearMethods.PCA(data = expression, out_dims=2)
    >>> type(results)
    <class 'CytofDR.dr.Reductions'>

If you have read the `Quick Start Guide <https://cytofdr.readthedocs.io/en/latest/quickstart.html>`_,
you may have notice that the object has a few built-in methods and atrributes, such as ``reductions``
that stores all the embeddings and ``evaluate`` used to evaluate the performance of all your DR methods. 

However, there are times when you may want to take advantage of this class but with your own
embeddings. For example, you may have other DR embeddings from other packages or custom
DR that we will detail below. In this case, you can create your own ``Reductions`` object:

.. code-block:: python

    >>> results = dr.Reductions(reductions = {"your_dr": embedding})

where ``embedding`` should be an array. Since ``eductions`` parameter accepts a dictionary, 
you can easily add multiple embeddings and name them however you like:

.. code-block:: python

    >>> results = dr.Reductions(reductions = {"your_fav_method": embedding1, "your_2nd_fav_method": embedding2})

This is simple enough!

Add New Embeddings
--------------------

If you would like to add new embeddings to an existing object, you can do that too! In fact,
it is allowed to create an empty object and add embeddings later using the ``add_reduction()``
methods:


.. code-block:: python

    >>> results = dr.Reductions()
    >>> results.add_reduction(reduction = embedding1, name = "your_dr")
    >>> results.add_reduction(reduction = embedding2, name = "your_dr2")
    >>> results.names
    ["your_dr", "your_dr2"]

This is a great way to integrate this framework into anywhere of your workflow. At the same time,
we allow you to use other DR methods along with our builtin methods to achieve maximum flexibility.


Add Metadata
---------------

When you use the ``run_dr_methods()`` wrapper, it automatically adds the expression
matrix to the resulting ``Reductions`` object even if you didn't notice it in the
first place:

.. code-block:: python

    >>> results = dr.run_dr_methods(expression, methods="PCA", out_dims = 3)
    >>> results.original_data
    array([[1.73462413, 2.44479204, 0.        , ..., 0.22536523, 1.02089248, 0.1500314 ],
           [0.56619612, 1.52259608, 0.        , ..., 0.31847633, 0.        , 0.        ],
           [0.54875404, 0.        , 0.        , ..., 0.17807296, 0.46455456, 3.55193468],
           ...,
           [0.1630427 , 0.32121831, 0.        , ..., 0.61940005, 0.        , 3.50253287],
           [0.30990439, 2.59020988, 0.11689489, ..., 0.94090453, 0.1383413 , 0.        ],
           [0.71138557, 1.72764796, 0.        , ..., 0.        , 0.        , 0.        ]])
           
which is exactly the expression matrix we showed at the beginning of this tutorial. However,
when you create your own ``Reductions`` object, this won't be done for you. But don't worry,
you can easily add it:

.. code-block:: python

    >>> results = dr.Reductions()
    >>> results.add_evaluation_metadata(original_data = expression)
    >>> results.original_data
    array([[1.73462413, 2.44479204, 0.        , ..., 0.22536523, 1.02089248, 0.1500314 ],
           [0.56619612, 1.52259608, 0.        , ..., 0.31847633, 0.        , 0.        ],
           [0.54875404, 0.        , 0.        , ..., 0.17807296, 0.46455456, 3.55193468],
           ...,
           [0.1630427 , 0.32121831, 0.        , ..., 0.61940005, 0.        , 3.50253287],
           [0.30990439, 2.59020988, 0.11689489, ..., 0.94090453, 0.1383413 , 0.        ],
           [0.71138557, 1.72764796, 0.        , ..., 0.        , 0.        , 0.        ]])

And voila, that's how you do it! Also, remember this method because if you want to do evaluations,
you may need this again to add your own clusterings and cell types as detailed in our
`Evaluation Metrics tutorial <https://cytofdr.readthedocs.io/en/latest/tutorial/metrics.html>`_.

----------------------------------

*******************
Run Your own DR
*******************

Since ``dr.run_dr_methods()`` is simply a wrapper, you can run each method on your
own. This offers a few advantages:

1. You can have greater flexibility with customization options.
2. You can decide when to run each method.

Each method can be accessed using methods listed in the table above. We have two
classes that house DR methods: ``LinearMethods`` and ``NonlinearMethods``. As their
names suggest, they have **linear** and **nonlinear** DR methods respectively. The methods
have the following common parameters:

=========== ================== ========================================= ==============
Parameter     Type               Meaning                                  Support
----------- ------------------ ----------------------------------------- --------------
data          ``np.ndarray``    The expression matrix                      All
out_dims      ``int``           The number of dimensions of the output     All
n_jobs        ``int``           The number of parallel jobs to run         Some
=========== ================== ========================================= ==============

If available, the default is to run the jobs on as many threads as possible, and the output
is 2D. Further, all methods return a ``numpy`` array. They are all static methods, meaning that
no initialization of instance objects is necessary. You can use them as functions.

To run a method, you can simply run:

.. code-block:: python

    >>> dr.LinearMethods.PCA(data = expression, out_dims=2)
    array([[ 3.95384698e+00, -5.18934256e-03],
           [ 4.67605078e+00, -1.34965157e+00],
           [-2.04514713e+00, -1.26489972e+00],
           ...,
           [-2.66635013e+00, -2.01899597e+00],
           [ 5.76069021e+00, -1.24300920e+00],
           [ 3.30609832e+00, -2.16666684e+00]])

All methods have similar interface: once you've learned one, you've pretty much good to go!


Customization
---------------

One advantage of running each method by yourself is that you can customize your DR. Most
methods, except ``dr.NonLinearMethods.open_tsne`` which will be the detailed in the
following sections, have the ``**kwargs`` option. This is passed directly to the original
implementation methods. For example, you can change a few key parameters for UMAP:

.. code-block:: python

    >>> dr.NonLinearMethods.UMAP(data = expression, out_dims=2, n_neighbors = 30, min_dist = 0)
    array([[-1.9451777, 10.632807 ],
           [-1.659229 , -3.771215 ],
           [ 8.913493 , -4.3567996],
           ...,
           [ 9.952888 , -4.4339495],
           [-1.4845752, -4.266201 ],
           [-2.748104 , -3.0372431]], dtype=float32)

**The one caveate** with customization is that you need to work with your own ``Reductions``
object! This gives you the freedom to tweak parameters to your liking or even tune
parameters at the expense of a few more lines of code. For example, you can run the following:

.. code-block:: python

    >>> embedding = dr.NonLinearMethods.UMAP(data = expression, out_dims=2, n_neighbors = 30, min_dist = 0)
    >>> results = dr.Reductions(reductions = {"custom_umap": embedding})

With this, you can go onto use your ``Reductions`` object as usual.


tSNE (``open_tsne``)
----------------------

The one exception to ``open_tsne``'s customization is that we don't allow ``**kwargs`` to avoid
ambiguity and confusion. This is the case because we internally utilzed the
`advanced framework <https://opentsne.readthedocs.io/en/latest/examples/02_advanced_usage/02_advanced_usage.html>`_,
which does not have a simple interface. Instead, we have provided our interface to key arguments.
This comes with some crucial advantages:

- There is support for multiple perplexities.
- Users can use custom initialization besides ``pca`` or ``spectral``.

Meanwhile, there are still great flexibities to change a few key parameters. For exact details, visit the
`Full API Reference <https://cytofdr.readthedocs.io/en/latest/documentation/dr.html>`_.


