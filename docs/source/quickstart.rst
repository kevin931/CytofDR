####################
Quickstart Guide
####################

You don't need to know every detail of every method to get CytofDR working. In fact, you don't even
need to know much Python. ``CytofDR`` is a flexible and extensible framework to allow you to perform
and benchmark dimension reduction all in one place with human readable codes. Here, there are examples
to walk you through every step of the way! Scroll to **Pipeline At a Glance** for **TLDR**.

-----------

****************
Loading Dataset
****************

The first step of the process is loading your CyTOF sample into Python. The ``CytofDR`` package uses the
``numpy`` framework extensively, which means that loading expression matrices are fairly easy. Here, we
assume that all your datasets have been preprocessed and only lineage channels are preserved. If you need
to preprocess your data, you can use our sister package `PyCytoData <https://github.com/kevin931/PyCytoData>`_.


Assume that you have a csv with the rows as cells and columns as features. Further, the first row is 
feature names, here is what you can do:

.. code-block:: python

    >>> import numpy as np
    >>> expression = np.loadtxt(fname="PATH_To_file", dtype=float, skiprows=1, delimiter=",")
    
Voila, you have an expression matrix in an array! You can view the array by simply calling it:

.. code-block:: python

    >>> expression

    array([[1.73462413, 2.44479204, 0.        , ..., 0.22536523, 1.02089248, 0.1500314 ],
           [0.56619612, 1.52259608, 0.        , ..., 0.31847633, 0.        , 0.        ],
           [0.54875404, 0.        , 0.        , ..., 0.17807296, 0.46455456, 3.55193468],
           ...,
           [0.1630427 , 0.32121831, 0.        , ..., 0.61940005, 0.        , 3.50253287],
           [0.30990439, 2.59020988, 0.11689489, ..., 0.94090453, 0.1383413 , 0.        ],
           [0.71138557, 1.72764796, 0.        , ..., 0.        , 0.        , 0.        ]])

.. note:: CytofDR does not support working with fcs files directly!
.. note:: For outputs, we will use the example of the Oejen cohort Sample U.

----------------------

*********************
Dimension Reduction
*********************

DR running DR is as easy as copy some code and hitting enter! Let's say you want to run UMAP, tSNE,
and PCA--three of the most popular methods. You can simply do the following:

.. code-block:: python

    >>> from CytofDR import dr
    >>> results = dr.run_dr_methods(expression, methods=["umap", "open_tsne", "pca"])

    Running PCA
    Runnign UMAP
    Running open_tsne
    ===> Finding 90 nearest neighbors using Annoy approximate search using euclidean distance...
    --> Time elapsed: 81.40 seconds
    ===> Calculating affinity matrix...
    --> Time elapsed: 2.84 seconds
    ===> Running optimization with exaggeration=12.00, lr=10243.67 for 250 iterations...
    Iteration   50, KL divergence 6.8745, 50 iterations in 2.1659 sec
    Iteration  100, KL divergence 6.3337, 50 iterations in 2.2041 sec
    Iteration  150, KL divergence 6.2017, 50 iterations in 2.3244 sec
    Iteration  200, KL divergence 6.1405, 50 iterations in 2.2421 sec
    Iteration  250, KL divergence 6.1041, 50 iterations in 2.2620 sec
    --> Time elapsed: 11.20 seconds
    ===> Running optimization with exaggeration=1.00, lr=10243.67 for 250 iterations...
    Iteration   50, KL divergence 4.8511, 50 iterations in 2.1616 sec
    Iteration  100, KL divergence 4.3954, 50 iterations in 2.1746 sec
    Iteration  150, KL divergence 4.1621, 50 iterations in 2.2973 sec
    Iteration  200, KL divergence 4.0129, 50 iterations in 2.5909 sec
    Iteration  250, KL divergence 3.9067, 50 iterations in 2.9614 sec
    --> Time elapsed: 12.19 seconds

We have some handy printouts to remind you what is running, but if you would like disable so that
it doesn't clutter your precious console screen, you can specify ``verbose=False``. 

Access Embeddings
----------------------

You can easily access the embeddings of that are stored in the object by accessing the ``reductions``
dictionary and use the method names as keys.

.. code-block:: python

    >>> results.reductions["UMAP"] 

    array([[-1.1084751 , 10.174761  ],
           [ 0.7808647 , -2.341636  ],
           [12.979893  , -5.1433287 ],
           ...,
           [11.690209  , -5.4123435 ],
           [ 0.9842613 , -2.8788142 ],
           [ 1.6086756 , -0.92493653]], dtype=float32)

To know the names of your embeddings, you can simply run:

.. code-block:: python

    >>> results.reductions.keys() 

    dict_keys(['PCA', 'UMAP', 'open_tsne'])

Plotting Results
-----------------

One of the main goals of DR is to visualize the data! Wanna know whether T cells are next to
B cells? We've got your back like your best friend! You can simply run the following:

.. code-block:: python

    results.plot_reduction("umap", save_path="PATH_To_FILE")

Here is an example of the embedding:

.. image:: ../../assets/ex_scatter.png
   :alt: scatter

Umm, something is missing! There're no labels: it looks a bit dull! If you have labels or
cell types, you can do so by specifying the ``hue`` parameter: 

.. code-block:: python

    ## ``labels`` is a numpy array of labels
    results.plot_reduction("umap", save_path="PATH_To_FILE", hue=labels)

Here are the results of colored clusters:

.. image:: ../../assets/ex_scatter_labels.png
   :alt: scatter_labels

Much better!

-----------------

*****************
DR Evaluation
*****************

Have you wondered which DR method is the best? Well, you can benchmark it yourself! This comes in two
steps! First, you will need to choose metrics and evaluate your DR methods! Then, you can rank your
methods according to these methods!

Currently, we do not support using custom methods for this framework. However, we have the following
categories of metrics:

- Global Structure Preservation ("global")
- Local Structure Preservation ("local")
- Downstream Performance ("downstream")
- Concordance ("concordance")

.. note:: The ``concordance`` category is more advanced! We will detail this more in the tutorial section.


Add Evaluation Metadata
-------------------------

Since many of the evaluation methods rely on additional information, we recommend have at least clusterings
for the original space data (expression matrices) and the embeddings! We will provide an interface to cluster
them in the future. However, if you already have the cluters ready, you can add them using the following method:

.. code-block:: python

    >>> results.add_evaluation_metadata(original_data = expression,
                                        original_labels = original_labels,
                                        embedding_labels = embedding_labels)


These are the **bare-minimum** needed! Here, ``original_data`` and ``original_labels`` are ``numpy`` arrays.
On the other hand, ``embedding_labels`` is a dictionary with name of DR methods as keys and ``numpy`` arrays
of labels as the values. You can, of course, load these data using the methods demonstrated above!

However, if you also have cell types:

.. code-block:: python

    >>> results.add_evaluation_metadata(original_data = expression,
                                        original_labels = original_labels,
                                        original_cell_types = original_cell_types,
                                        embedding_labels = embedding_labels,
                                        embedding_cell_types = embedding_cell_types)

which will allow you to run **Cell Type-Clustering Concordace** metrics as part of the ``downstream`` category. Here,
``original_cell_types`` is just a ``numpy`` array, whereas ``embedding_cell_types`` is a dictionary.


Run Evaluation
----------------

Once you have all your metadata loaded into the object, you can simply do the following:

.. code-block:: python
    
    >>> results.evaluate(category = ["global", "local", "downstream"])

    Evaluating global...
    Evaluating local...
    Evaluating downstream...

Even if you don't have ``embedding_cell_types`` and ``original_cell_types``, the method will adjust accordingly.
You can access the metrics all at once:

.. code-block:: python

    >>> results.evaluations

    {'global': {'spearman': {'PCA': 0.4895579809742763, 'UMAP': 0.1350164421924906, 'open_tsne': 0.39192577549108076},
    'emd': {'PCA': 2.2530293870587297, 'UMAP': 3.239140089959797, 'open_tsne': 25.696447153678164}},
    'local': {'knn': {'PCA': 0.0005857277667501871, 'UMAP': 0.002525137483323029, 'open_tsne': 0.004472682307767401},
    'npe': {'PCA': 1105.9800000000002, 'UMAP': 675.9279999999999, 'open_tsne': 716.3840000000001}},
    'downstream': {'cluster reconstruction: silhouette': {'PCA': 0.028794289043266457, 'UMAP': 0.31541705, 'open_tsne': 0.1715173434882497},
    'cluster reconstruction: DBI': {'PCA': 6.088182130120897, 'UMAP': 2.682592524176189, 'open_tsne': 2.1286075869918295},
    'cluster reconstruction: CHI': {'PCA': 51150.88086144542, 'UMAP': 146647.8329712303, 'open_tsne': 48843.649928793435},
    'cluster reconstruction: RF': {'PCA': 0.6565512141008258, 'UMAP': 0.922223591766301, 'open_tsne': 0.9340811044003451},
    'cluster concordance: ARI': {'PCA': 0.42098863850027674, 'UMAP': 0.687537599275774, 'open_tsne': 0.46566317172488875},
    'cluster concordance: NMI': {'PCA': 0.5708888916428192, 'UMAP': 0.7700788594536911, 'open_tsne': 0.6756652682496738},
    'cell type-clustering concordance: ARI': {'PCA': 0.20987452324575112, 'UMAP': 0.3269339854687899, 'open_tsne': 0.1472176998953652},
    'cell type-clustering concordance: NMI': {'PCA': 0.30133272742067724, 'UMAP': 0.44913037545258316, 'open_tsne': 0.35246029727387274}}}

This is a nested dictionary with the following levels:

1. Categories
2. Metrics/Sub-categories
3. Embedding Names

This can be a little confusing, but you can access the sub-levels individually:

.. code-block:: python

    >>> results.evaluations["global"]

    {'spearman': {'PCA': 0.4895579809742763, 'UMAP': 0.1350164421924906, 'open_tsne': 0.39192577549108076},
    'emd': {'PCA': 2.2530293870587297, 'UMAP': 3.239140089959797, 'open_tsne': 25.696447153678164}}


or you can look at individual metrics:

.. code-block:: python
    
    >>> results.evaluatios["global"]["emd"]

    {'PCA': 2.2530293870587297, 'UMAP': 3.239140089959797, 'open_tsne': 25.696447153678164}

If you are so inclined, you can utilize these results directly. However, if you would like us to do the work for you,
read on!


Rank DR Methods
-------------------

Now, you can finally rank your methods! This will be fairly easy:

.. code-block:: python

    >>> results.rank_dr_methods()

    {'PCA': 1.8055555555555556, 'UMAP': 2.277777777777778, 'open_tsne': 1.9166666666666667}

As you can see, this returns a dictonary with method names the methods as keys and their scores
as values. If you see decimals, don't panic! at your computer! We rank each metric
individually and the final results are appropriately weighted! Here, larger score is
better! Obviously, if you have read `our paper <https://www.biorxiv.org/content/10.1101/2022.04.26.489549v1.abstract>`_,
you know that UMAP is pretty good at what it does when compared to PCA and tSNE! 


----------------

**********************
Pipeline At a Glance
**********************

Putting everything together, we will have a pipeline like this:

.. code-block:: python

    >>> from CytofDR import dr

    >>> results = dr.run_dr_methods(expression, methods=["umap", "open_tsne", "pca"])
    >>> results.add_evaluation_metadata(original_data = expression,
    ...                                 original_labels = original_labels,
    ...                                 original_cell_types = original_cell_types,
    ...                                 embedding_labels = embedding_labels,
    ...                                 embedding_cell_types = embedding_cell_types)
    >>> results.evaluate(category = ["global", "local", "downstream"])
    >>> results.rank_dr_methods()


Congratulations! You've made it through the quickstart guide! Give yourself a high five
and start performing DR! For more detailed documentations, look around on this website!


**********************
What Next?
**********************


