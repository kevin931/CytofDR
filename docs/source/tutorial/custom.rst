########################
Custom DR Evaluations
########################

Our existing DR evaluation framework has four major categories, which is a great way
to evaluate your DR methods by simplying use some or all of them. However, sometimes
it makes sense to implement your own metric. For example, you may only want EMD but
not correlation. Or, you would like to different options for some metrics. This
tutorial is for you: we will show you how you can add your own results and rank them.

Before we begin, we will be using the ``levine13`` dataset for demonstration purposes,
which can be loaded in the following way:

.. code-block:: python

    >>> from PyCytoData import DataLoader
    >>> from CytofDR.dr import run_dr_methods
    >>> exprs = DataLoader.load_dataset(dataset="levine13")
    >>> exprs.preprocess(arcsinh=True)
    >>> results = run_dr_methods(data=exprs.expression_matrix, methods=["PCA", "ICA"])
    Running PCA
    Running ICA

--------------------------------

*********************************
Working with Your Own Metrics
*********************************

First, you will need to run the metrics to get the results. If you have implemented your
own metrics, then you can skip to the next section and add your metrics. Otherwise, you
can use the metrics that are built into this package: a complete list of such metrics
can be found `in this tutorial <https://cytofdr.readthedocs.io/en/latest/tutorial/metrics.html>`_.

Suppose you would like to evaluate your DR methods with **Neighborhood Agreement**, which
is not included in our default DR evaluation framework. You will need to run it manually
which is not really hard to do! **Remember**: you will need to evaluate every method in
your ``Reductions`` class for the subsequent ranking to work properly. Now, to run these
two metrics, we first to need get some things set up. Specifically, we need to make the
necessary import, compute neighbors for original and embeddings, and then have ways to
capture the results.

.. code-block:: python

    >>> from CytofDR.evaluation import EvaluationMetrics as EM

    >>> original_neighbors = EM.build_annoy(exprs.expression_matrix, k=100)
    >>> agreement = []

Now, we are going to loop through each embedding and find their neighbors and then
evaluate the metrics:

.. code-block:: python

    >>> for reduction in results.reductions.keys():
    ...     embedding_neighbors = EM.build_annoy(results.reductions[reduction])
    ...     agreement.append(EM.neighborhood_agreement(original_neighbors, embedding_neighbors))
    >>> agreement
    [-0.0003080110625877093, -0.00031328229128209546]
    >>> results.add_custom_evaluation_result(metric_name = "Neighborhood Agreement", reduction_name = "PCA", value = agreement[0])
    >>> results.add_custom_evaluation_result(metric_name = "Neighborhood Agreement", reduction_name = "ICA", value = agreement[1])

Now that you have added your results, you can rank your DR method as usual:

.. code-block:: python

    >>> results.rank_dr_methods_custom()
    {'ICA': 1.0, 'PCA': 2.0}

**You have a ranking!** You may realize that this example is trivial for the following reasons:

1. There are only two DR methods.
2. There is only one metrics.

You can just look at the results and determine the ranking, but you can easily extend this to
more methods by including everything in the for loop:

.. code-block:: python

    >>> for reduction in results.reductions.keys():
    ...     embedding_neighbors = EM.build_annoy(results.reductions[reduction])
    ...     agreement = EM.neighborhood_agreement(original_neighbors, embedding_neighbors)
    ...     results.add_custom_evaluation_result(metric_name = "Neighborhood Agreement", reduction_name = reduction, value = agreement)


---------------------------

**************************************
Metrics with Custom Weighting Scheme
**************************************

By default, all your custom metrics are weighted equally: the ranks will be averaged for
all metrics. This works most of the times, but like in the case of our default evaluation
schemes, issues can arise. For example, if you decide to use two neighborhood-based metrics
and one cluster-based metric, your evaluation may become a bit unbalanced with the former
two outweighing the latter. In this case, you can assign different weights.

Doing so is fairly easy! We will use the ``results`` object from the intro section of this
tutorial. Suppose you would like to use ``NPE``, ``KNN``, and ``Neighborhood Agreement``,
and you want to weight ``NPE`` to consist of 50% of the weight, which makes sense because
``KNN`` and ``Neighborhood Agreement`` are equivalent. Thus, you can do:

.. code-block:: python

    >>> for reduction in results.reductions.keys():
    ...     embedding_neighbors = EM.build_annoy(results.reductions[reduction])
    ...     agreement = EM.neighborhood_agreement(original_neighbors, embedding_neighbors)
    ...     knn = EM.KNN(original_neighbors, embedding_neighbors)
    ...     npe = EM.knn(original_neighbors, embedding_neighbors)
    ...     results.add_custom_evaluation_result(metric_name = "Neighborhood Agreement", reduction_name = reduction, value = agreement, weight = 0.25)
    ...     results.add_custom_evaluation_result(metric_name = "NPE", reduction_name = reduction, value = knn, weight = 0.5, reverse_ranking = True)
    ...     results.add_custom_evaluation_result(metric_name = "KNN", reduction_name = reduction, value = npe, weight = 0.25)

If you're familiar with our default framework, you may notice that we're essentially
implementing a sub-category.

One thing to note is that we currently **do not** require all weights to add to 1. Thus, you
need to take some care not to implement nonsensical weighting schemes.


------------------------------

***********************
Reverse Ranking
***********************

You may have noticed the following line of code in the example from the previous
section:

.. code-block:: python

    >>> results.add_custom_evaluation_result(metric_name = "NPE", reduction_name = reduction, value = knn, weight = 0.5, reverse_ranking = True)

Here, we set ``reverse_ranking = True``. The reason we do so is that smaller value
for ``NPE`` is better. By default, our ranking system ranks our metrics by assigning
higher ranks to larger values. But when a metric's meaning is reversed, we want to
reverse the ranking procedure by **assigning higher ranks to smaller values**. The
following builtin metrics from the ``EvaluationMetrics`` class need to be reverse
ranked:

- ``EMD``
- ``NPE``
- ``davies_bouldin``
- ``embedding_concordance``
- ``residual_variance``

Once the parameter is set, the ``rank_dr_methods_custom`` will take care of
everything accordingly.
