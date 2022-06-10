IO and Preprocessing with PyCytoData
================================================

If you want to preprocess your CyTOF dataset befoe running DR, fear not! We have
you covered. We are also the developer of ``PyCytoData``, which is focused on
IO and preprocessing for CyTOF experiments. Further, it allows us to use a single
pipeline for everything. This tutorial showcases how we can utilize this pipeline
for a DR-focused project.

Please also feel free to read more in-depth documentation on ``PyCytoData``'s
`Official Documentation <https://pycytodata.readthedocs.io/en/latest/>`_.


--------------------

***************************
Loading Benchmark Datasets
***************************

Previously in the `Quick Start Guide <https://cytofdr.readthedocs.io/en/latest/quickstart.html>`_,
we've showcased how to load datasets with ``numpy``, which is very easy. However, if you want
to work with a few famous benchmark datasets, such as ``levine13`` and ``levine32``,
``PyCytoData`` offers an easy solution to help you achieve that goal:

.. code-block:: python

    >>> from PyCytoData import DataLoader
    >>> exprs = DataLoader.load_dataset(dataset = "levine13")
    Would you like to download levine13? [y/n]y

    Download in progress...
    This may take quite a while, go grab a coffee or cytomulate it!

And you have successfully download the ``levine13`` dataset. The dataset is automatically
cached so that you don't have to repeatedly download it every time you use it. You can
access the the expression matrix easily:

.. code-block:: python

    >>> exprs.expression_matrix
    array([[ 5.75381927e+01,  1.21189880e+01,  2.75074673e+00, ...,
             2.60543274e+02,  1.54974432e+01,  8.29685116e+00],
           [ 8.16322708e+01,  2.34020500e+01,  1.57276118e+00, ...,
             1.75833466e+02,  2.17522359e+00,  3.34277302e-01],
           [ 2.10737019e+01,  4.41922474e+00, -5.81668496e-01, ...,
             2.27592499e+02,  6.24691308e-01, -1.94343376e+01],
           ...,
           [ 1.59633112e+01,  9.53633595e+00,  4.49561157e+01, ...,
             3.46169220e+02,  2.27766180e+00,  4.33450623e+01],
           [ 2.25081215e+01,  8.42314911e+00,  8.56426620e+01, ...,
             6.43495300e+02,  5.97545290e+00,  8.84256649e+00],
           [ 2.82463398e+01,  7.47339916e+00,  5.64270020e+01, ...,
             6.65499023e+02, -7.26899445e-01,  7.11599884e+01]])

From here, you can use the expression matrix to do everything you need to
do in ``CytofDR``. 

We have the following datasets available:

============== ==========
Dataset Name    Literal
-------------- ----------
Levine-13dim    levine13
Levine-32dim    levine32
Samusik         samusik
============== ==========

Currently, they have mostly been preprocessed, except for ``Acrsinh`` transformation,
which we will detail below.

*************************
Loading Your Own Dataset
*************************

Of course, you don't have to use a benchmark dataset! You can use your
own dataset:

.. code-block:: python

    >>> from PyCytoData import FileIO
    >>> exprs = FileIO.load_delim(dataset = "PATH_TO_EXPRS", col_names=True, delim="\t")
    >>> type(exprs)
    <class 'PyCytoData.data.PyCytoData'>


This is very reminiscent of ``numpy`` approach or the ``R`` approach if you're familiar with it.
Here, we assume that the data is stored in plain text, deliminated file. Rows are cells and columns
are features. If ``col_names=True``, then the first row is treated as channel names. And again,
this is a ``PyCytoData`` object, and you can access its ``expression_matrix`` for all your DR needs.

-------------------------

Preprocessing
--------------

Once you have a ``PyCytoData`` object such as the ones we've created above, preprocessing is
really just one line of code away. We offer the following preprocessing steps:

- Arcsinh transformation
- Gate to remove derbis
- Gate for intact cells
- Gate for live cells
- Gate using Center, Offset, and Residual channels
- Bead normalization

And you can pick and choose which of these steps to apply to your particular dataset. For
benchmark datasets, all you need to do is this:

.. code-block:: python

    >>> exprs.preprocess(arcsinh=True)
    Runinng Arcsinh transformation...

Now, you can accessed you preprocessed expression matrix:

.. code-block:: python

    >>> exprs.expression_matrix()
    array([[ 4.05275087,  2.50151373,  1.12358426, ...,  5.5627837 ,
             2.74481299,  2.13009628],
           [ 4.40237469,  3.15464461,  0.72199792, ...,  5.16956967,
             0.94198797,  0.16637009],
           [ 3.05027008,  1.53363094, -0.28688286, ...,  5.42757605,
             0.30747774, -2.96967868],
           ...,
           [ 2.77419437,  2.26592833,  3.80618123, ...,  5.84693608,
             0.97621692,  3.76972462],
           [ 3.11584426,  2.14478932,  4.45031986, ...,  6.46691714,
             1.81455806,  2.19212776],
           [ 3.34221489,  2.02879191,  4.03326172, ...,  6.50053943,
            -0.35588934,  4.26512812]])

For your own dataset, you can run the whole suite if you like: 

.. code-block:: python

    >>> exprs.preprocess(arcsinh=True,
    ...                  gate_debris_removal=True,
    ...                  gate_intact_cells=True,
    ...                  gate_live_cells=True,
    ...                  gate_center_offset_residual=True,
    ...                  bead_normalization=True)
    Runinng Arcsinh transformation...
    Runinng debris remvoal...
    Runinng gating intact cells...
    Runinng gating live cells...
    Runinng gating Center, Offset, and Residual...
    Runinng bead normalization...

-----------------------------

****************************
Using CytofDR in PyCytoData
****************************

In the tutorial above, we've showcased how to extract the expression matrix and
then work with ``CytofDR``. This works perfectly, but you may wonder whether it's
possible to stay within the ``PyCytoData`` object. The answer is of course yes!
We've provided the ``run_dr_methods`` interface to ``PyCytoData``, but you can
also store a ``Reductions`` object within your ``PyCytoData`` object. This 
section will show you how to do so.

Quick DR with ``run_dr_methods``
----------------------------------

Once you have a ``PyCytoData`` object, you can simply run the method (here, we
will keep using the object created in the tutorials above):

.. code-block:: python

    >>>  exprs.run_dr_methods(methods = ["PCA", "UMAP", "ICA"])
    Running PCA
    Running ICA
    Running UMAP
    >>> type(exprs.reductions)
    <class 'CytofDR.dr.Reductions'>
    >>> exprs.reductions.evaluate(category=["Global"])
    Evaluating global...
    >>> exprs.rank_dr_methods()
    {'PCA': 1.5, 'ICA': 2.0, 'UMAP': 2.5}

This, really, is just a wrapper for the ``CytofDR`` version to allow you to
run DR directly. Further, the ``reductions`` attribute stores a ``Reductions``
object, meaning that once you've run your DR, you can use any ``Reductions``
object features and workflows as usual.

.. note::

    There is one significant **caveat** to note here: the ``transform`` option is
    not implemented here because of the ambiguity that it may cause. This may be
    included in a future feature update.

Using Your Own ``Reductions`` Object
--------------------------------------

As you may wonder whether you can do DR separately in ``CytofDR`` with more
features while still using ``PyCytoData``, the answer is you can. You can
store your own ``Reductions`` object in the ``PyCytoData`` object:

.. code-block:: python

    from CytofDR import dr

    >>> results = dr.Reductions()
    >>> results.add_reduction(reduction = embedding1, name = "your_dr")
    >>> results.add_reduction(reduction = embedding2, name = "your_dr2")
    >>> exprs.reductions = results

This effectively combines two objects into one! Now, you can proceed as you wish!






