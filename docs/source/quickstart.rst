####################
Quickstart Guide
####################

You don't need to know every detail of every method to get CytofDR working. In fact, you don't even
need to know much Python. ``CytofDR`` is a flexible and extensible framework to allow you to perform
and benchmark dimension reduction all in one place with human readable codes. Here, there are examples
to walk you through every step of the way!

-----------

****************
Loading Dataset
****************

The first step of the process is loading your CyTOF sample into Python. The ``CytofDR`` package uses the
``numpy`` framework extensively, which means that loading expression matrices are fairly easy.


Assume that you have a csv with the rows as cells and columns as features. Further, the first row is 
feature names, here is what you can do:

.. code-block:: python

    import numpy as np

    expression = np.loadtxt(fname="PATH_To_file", dtype=float, skiprows=1, delimiter=",")
    
Voila, you have an expression matrix in an array! You can go onto perform DR!

.. note:: CytofDR does not support working with fcs files directly!

----------------------

*********************
Dimension Reduction
*********************

DR running DR is as easy as copy some code and hitting enter! Let's say you want to run UMAP, tSNE,
and PCA--three of the most popular methods. You can simply do the following:

.. code-block:: python

    from CytofDR import dr

    results = dr.run_dr_methods(expression, methods=["umap", "open_tsne", "pca"])


To get the embeddings
----------------------

You can easily access the embeddings of that are stored in the object by accessing the ``reductions``
dictionary and use the method names as keys.

.. code-block:: python

    results.reductions["umap"]
    results.reductions["open_tsne"]
    results.reductions["pca"]


Plotting Results
-----------------

One of the main goals of DR is to visualize the data! Wanna know whether t cells are next to
B cells? We've got your back like your best friend! You can simply run the following:

.. code-block:: python

    results.plot_reduction("umap", save_path="PATH_To_FILE")

Here is an example of the embedding:

.. image:: ../../assets/ex_scatter.png
   :alt: Logo

Umm, something is missing! There're no labels: it looks a bit dull! If you have labels or
cell types, you can do so by specifying the ``hue`` parameter: 

.. code-block:: python

    ## ``labels`` is a numpy array of labels
    results.plot_reduction("umap", save_path="PATH_To_FILE", hue=labels)

Here are the results of colored clusters:

.. image:: ../../assets/ex_scatter_labels.png
   :alt: Logo

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

    results.add_evaluation_metadata(original_data = expression,
                                    original_labels = original_labels,
                                    embedding_labels = embedding_labels)


These are the **bare-minimum** needed! Here, ``original_data`` and ``original_labels`` are ``numpy`` arrays.
On the other hand, ``embedding_labels`` is a dictionary with name of DR methods as keys and ``numpy`` arrays
of labels as the values. You can, of course, load these data using the methods demonstrated above!

However, if you also have cell types:

.. code-block:: python

    results.add_evaluation_metadata(original_data = expression,
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
    
    results.evaluate(category = ["global", "local", "downstream"])

Even if you don't have ``embedding_cell_types`` and ``original_cell_types``, the method will adjust accordingly.
You can access the metrics with the following attributes:

.. code-block:: python

    results.evaluations # A dictionary of all metrics
    results.evaluations["global"] # A dictionary of a specific category with all its metrics
    results.evaluatios["global"]["emd"] # The value of a specific metric


Rank DR Methods
-------------------

Now, you can finally rank your methods! This will be fairly easy:

.. code-block:: python

    results.rank_dr_methods()

This will return a dictonary with method names the methods as keys and their scores
as values. If you see decimals, you panic! at your computer! We rank each metric
individually and the final results are appropriately weighted! Here, larger score is
better!


----------------

**********************
Pipeline At a Glance
**********************

Putting everything together, we will have a pipeline like this:

.. code-block:: python

    from CytofDR import dr

    results = dr.run_dr_methods(expression, methods=["umap", "open_tsne", "pca"])
    results.add_evaluation_metadata(original_data = expression,
                                    original_labels = original_labels,
                                    original_cell_types = original_cell_types,
                                    embedding_labels = embedding_labels,
                                    embedding_cell_types = embedding_cell_types)
    results.evaluate(category = ["global", "local", "downstream"])
    results.rank_dr_methods()


Congratulations! You've made it through the quickstart guide! Give yourself a high five
and start performing DR! For more detailed documentations, look around on this website!
