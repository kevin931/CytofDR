Evaluation Metrics
====================

One of the most important aspects of DR benchmarking is choosing a set of evaluation metrics.
If you are just looking to choose a DR method for your analyses, you can either use our
`guidelines <>`_ or be more careful and benchmark yourself. As our own experiences went, some
samples really benefit from careful selection of methods. Here, we will walk you through the
categories of evaluation metrics and how you can effectively use them in your workflow. 

In this section, we will again be utilizing the same dataset from the Oetjen cohort, but we will
need some metadat to work with. 

---------------------------------

****************************
Evaluation Metrics Overview
****************************

If you are interested in an overview, this table is designed for you:

=================================== ===================================================== ============== =========================== ===========================
Metric Name                            Method                                               Acronym         Category                   Sub-Category    
----------------------------------- ----------------------------------------------------- -------------- --------------------------- ---------------------------
Spearman's Correlation                ``EvaluationMetrics.correlation``                     COR             Global                       None
Earth Mover's Distance                ``EvaluationMetrics.EMD``                             EMD             Global                       None
K-Nearest Neighbors                   ``EvaluationMetrics.KNN``                             KNN             Local                        None
Neighborhood Proportion Eror          ``EvaluationMetrics.NPE``                             NPE             Local                        None
Silhouette Score                      ``EvaluationMetrics.silhouette``                      SIL             Downstream                   Cluster Reconstruction
Davies-Bouldin Index                  ``EvaluationMetrics.davies_douldin``                  DBI             Downstream                   Cluster Reconstruction
Calinski-Harabasz Index               ``EvaluationMetrics.calinski_harabasz``               CHI             Downstream                   Cluster Reconstruction
Random Forest Cluster Prediction      ``EvaluationMetrics.random_forest``                   RF              Downstream                   Cluster Reconstruction
Adjusted Rand Index                   ``EvaluationMetrics.ARI``                             ARI             Downstream/Concordance       Multiple
Normalized Mutual Information         ``EvaluationMetrics.NMI``                             NMI             Downstream/Concordance       Multiple
Embedding Concordance                 ``EvaluationMetrics.embedding_concordance``           None            Concordance                  Multiple
Residual Variance                     ``EvaluationMetrics.residual_variance``               None            Extra                        None
Neighborhood Agreement                ``EvaluationMetrics.neighborhood_agreement``          None            Extra                        None
Neighborhood Trustworthiness          ``EvaluationMetrics.neighborhood_trustworthiness``    None            Extra                        None
=================================== ===================================================== ============== =========================== ===========================

As you can see, we have implemented 15 different methods in ``evaluation.EvaluationMetrics`` class as
static methods. However, there are actually more! If you are keen-eyed and have looked around in our
paper of the Full API Reference, you can fit a few examples:

- We actually support both Spearman's and Pearson's correlation (although we recommend the former).
- There are two methods in ``EvaluationMetrics.embedding_concordance``: ``cluster_distance`` and ``emd``.
- ARI and NMI and used in multiple ways in Downstream and Concordance.

Don't worry: we will explain these quirks and features more in the following sections. However, these
methods are more general purpose than specific. If you are interested in using these implementations,
feel free to do so!

--------------------------------

**************************
Our Evaluation Framework
**************************

In our benchmark study, we constructed a comprehensive evaluation framework with four major categories,
which are also implemented in this package using metrics above! Here, we will walk you through all
the categories and their sub-categories for you to be confident in choose the category that is right
for you!

Global Structure Preservation (``global``)
--------------------------------------------

While running DR, we want the embedding to reflect the overall structure of the original data. Think
of it this way: if I were ask you to sketch a floorplan of your house, you are likely to focus on the
overall layout, which will tell me where the kitchen is and how the house is laid out. This is exactly
the idea of the global category. While you may not get the details, it shows whether we can trust
the relationships between cell types. 

In this category, there are two equally weighted metrics: ``COR`` and ``EMD``. You may ask: what're
the inputs? Great question! One thought is to use pairwise distance between all cells in the original
and the embedding space, but the CyTOF sample size makes calculating and storing the matrix impractical.
To remedy this, we use the Point Cluster Distance, which utilizes pairwise distance but between
individual cells and cluster centroid (more on this `here <>`_).


Local Structure Preservation (``local``)
--------------------------------------------

Sometimes, we need to know the details. Using the floor plan example above, we may want to know
the color of your wall paper or whether have a coffee machine on the counter. As you notice,
these are much more detailed and usually doesn't appear on the floor plan. And for CyTOF DR
evaluation, we want to exactly take this into consideration. Here, we want to know more about
neighors. Are neighbors in the original space still neighbors? Are a cell's neighbors the same
type of cells or of different types of cells?

If you can answer  the above questions, you already the metrics in your mind! The first question
can be answered to use **KNN**. Simply, we find the neighbors for each cell before and after DR, and
we find the overlap and average it! Of course, the overlap will be 1 if neighbors don't change at
all, and that'll be ideal! To answer the second question, we will need cell type information (if 
you don't have this, you will need to use clustering). Then, using the neighbor graph, we find
the average proportion of cells in the neighborhood that don't belong to the same type! Usually,
cells of the same type should be more similar, and we expect ``NPE`` to be small!


Downstream Analysis Performance (``downstream``)
--------------------------------------------------

Besides visualizing your CyTOF data, performing DR can also help you perform other downstream
analyses. For example, if you have the floor plan, I want to know how well your furniture can
fit into the house! If they fit the space well and are useful, then we can say the downstream
performance is good; on the other hand, if they're awkward (e.g. you will have to move the dining
table to get from the living room to the kitchen), then of course it's not as convenient. This
kind of idea applies in CyTOF! If we can't do anything with embeddings, then we will start
questioning the purpose of DR in the first place. Thus, we want to benchmark the performance 
of downstream analyses by using each of the DR methods.

This category is perhaps the most complicated of them all! There are three equally weighted
sub-categories. The first is **Cluster Restruction**. As its name suggests, its metrics,
``SIL``, ``DBI``, ``CHI``, and ``RF`` measures how an embedding can reconstruct clustering
done using the original data. If DR loses little information, then it should faithly
recreate clusters of the original space. Thus, we use these four metrics to assess how the 
quality of such clusters. The first three metrics are quite convention, but ``RF``, which 
is based on random forest is a bit different. We first train a random forest classifier
using a training set from the embedding, and test how well we can predict the cluster
labels using the testing set. Of course, we want the accuracy to be high in this case.

The second category is **Cluster Concordance**, in which we cluster both in the original space
and the embedding space, and then we compute ``ARI`` and ``NMI`` to assess whether
the two clustering results agree. The third category is **Cell Type-Clustering Concordance**.
It utilized the same metrics, ``ARI`` and ``NMI``, but it compares cell types found from
the original data with embedding space clusterings. This allows us to combine both practial
cell typing and clusterings into one sub-category.

Concordance
-------------

This is the last category of the evaluation framework. Since these days single cell
technologies are more and more common, we want to investigaet whether results from
different technologies will yield concordant results when we use the same DR method.
Continuing with the same example, let's say you would like to fix your heating system
in the house, and you know two good technicians in town. You ask each of for a quote,
and ideally, what they come up with should be concordant in terms of prices and plans.
If they are way off, then you can sense something is not right. Here, we are looking
at the same idea! The relationships between cell types should be preserved regardless
of what technologies people use.

To capitalize on the ideas, we develped Cluster Distance and EMD by considering
the rank distance between cell type centroids. These metrics allow us to assess
whether the relationship between cells have changed. Further, we implemented
**Gating Concordance**, which consists of ``ARI`` and ``NMI`` between cell types
of the original space and cell types of the embedding space. This category is mainly a
validation metric because this who category relies on the idea that the cell typing
information makes sense.

----------------------------------

*******************
Ranking DR Methods
*******************

One thing you may have noticed is that our metrics all have different interpretations 
and, more importantly, different scales. This means that we need to rank them
so that we can know  what's the best. The next challenge we face is: How do we
ranking them? How should we weight each metric?

This section deals with exactly these issues. There are a few principles we need
to follow:

1. All categories should have the same weight.
2. Within each category, its sub-category should have the same weight.
3. We need to reverse rank methods when a metric's value is smaller better.
4. When there are ties, we need to deal with them consistently. 

In our case, we define ranks to such that higher is better! In other words, those who
rank the top will have higher rank than other methods. Also, when there ties, we
emplot the "Max" method. In other words, when multiple methods have the same value,
we assign the maximum rank the three of them can normally take.

The Weighting Scheme
-----------------------

For all individual metrics, we rank the methods first! Then, to satisfy **Principle 1**
and **Principle 2**, we need to come up a weighting scheme. To do this, we follow
the following steps:

- We average the ranks within the sub-categories if applicable.
- We average the averaged ranks of sub-categories to form the scores of the category.
- Then, we average the averaged scores from all categories for the final score.

In other words, we take a weighted average on the ranks to account for the categories
and sub categories. The following table is a summary of all the metrics and their weights:


========================== =================================== ============ =============== =============
 Category                   Sub-Category                        Metric        Weight          Formula
-------------------------- ----------------------------------- ------------ --------------- -------------
 Global                       None                               COR            0.125         1/4/2
 Global                       None                               EMD            0.125         1/4/2
 Local                        None                               KNN            0.125         1/4/2
 Local                        None                               NPE            0.125         1/4/2
 Downstream                   Cluster Reconstruction             SIL            0.0208        1/4/3/4
 Downstream                   Cluster Reconstruction             DBI            0.0208        1/4/3/4
 Downstream                   Cluster Reconstruction             CHI            0.0208        1/4/3/4
 Downstream                   Cluster Reconstruction             RF             0.0208        1/4/3/4
 Downstream                   Clsuter Concordance                ARI            0.0417        1/4/3/2
 Downstream                   Clsuter Concordance                NMI            0.0417        1/4/3/2
 Downstream                   Cell Type-Clustering Concordance    ARI            0.0417        1/4/3/2
 Downstream                   Cell Type-Clustering Concordance    NMI            0.0417        1/4/3/2
 Concordance                  Cluster-Distance                   CCD            0.0833        1/4/3
 Concordance                  EMD                                EMD            0.0833        1/4/3
 Concordance                  Gating Concordance                 ARI            0.0417        1/4/3/2
 Concordance                  Gating Concordance                 NMI            0.0417        1/4/3/2
========================== =================================== ============ =============== =============

If you use all the metrics of our evaluation framework, the weight column will be exact. However,
if your own evaluation framework differs slight, all you need to do is to understand the formula.
Starting with 1, we first divide by the number of categories, then the number of sub-categories in
each category, and finally the number of metrics with each sub-categories. When there are no
sub-categories, you can treat the individual metrics as categories.

Now, you too can benchmark DR like a pro!

----------------------------------------

******************************************
Hands-On: Evaluating and Ranking Your DR
******************************************

With all the knowledge, you can evaluate and rank your methods. If you stick with the default pipeline,
everything will very easy! But before we begin, we will need the following metadata aside from the
expression matrix: 

.. code-block:: python

    >>> original_labels

    array(['20', '22', '4', ..., '4', '22', '22'], dtype='<U2')

    >>> original_cell_types

    array(['CD8T', 'CD4T', 'unassigned', ..., 'unassigned', 'CD4T', 'CD4T'], dtype='<U11')

    >>> embedding_labels

    {'PCA': array(['18', '23', '4', ..., '4', '23', '19'], dtype='<U2'),
     'UMAP': array(['23', '22', '7', ..., '7', '22', '22'], dtype='<U2'),
     'ICA': array(['12', '7', '4', ..., '4', '7', '1'], dtype='<U2')}

    >>> embedding_cell_types

    {'PCA': array(['CD4T', 'CD4T', 'unassigned', ..., 'unassigned', 'CD4T', 'CD4T'], dtype='<U10'),
     'UMAP': array(['CD8T', 'CD4T', 'unassigned', ..., 'unassigned', 'CD4T', 'CD4T'], dtype='<U11'),
     'ICA': array(['CD4T', 'CD4T', 'NK', ..., 'NK', 'CD4T', 'CD8T'], dtype='<U10')}

You will need the clustering for the original and DR space withe ``original_labels`` and ``embedding_labels``.
These two are mandatory. Optionally, you can also add original space cell types for **Cell Type-Clustering Concordance**,
which we will detail later.

.. note:: 

    Notice that ``embedding_labels`` is a dictionary because we need clusterings based on each DR method.

**Caveat**: Of course, we don't have labels of embeddings ahead of the time
because we are performing DR right here! We will include clustering capabilities in this package
in the future. For now, you will have to cluster using another package of your choice!


Example with Cell Types
-------------------------

Suppose you have a ``Reductions`` object with DR already performed:

.. code-block:: python

    >>> type(results)

    <class 'CytofDR.dr.Reductions'>

    >>> results.reductions.keys

    dict_keys(['PCA', 'ICA', 'UMAP'])

Then, you can add your metadat and proceed to evaluate your methods:

.. code-block:: python

    >>> results.add_evaluation_metadata(original_data = expression,
    ...                                 original_labels = original_labels,
    ...                                 original_cell_types = original_cell_types,
    ...                                 embedding_labels = embedding_labels)
    >>> results.evaluate(category = ["global", "local", "downstream"])

    Evaluating global...
    Evaluating local...
    Evaluating downstream...

    >>> results.rank_dr_methods()

    {'PCA': 1.9722222222222223, 'ICA': 1.5277777777777777, 'UMAP': 2.5}

As you can see, the methods are successfully evaluated and ranked! As expected,
UMAP is the best of the three. This is quite easy! 


Examples Without Original Space Cell Types
-------------------------------------------

In this case, you won't be adding cell types because you don't have them. But the overall
procedure is the same:

.. code-block:: python

    >>> results.add_evaluation_metadata(original_data = expression,
                                        original_labels = original_labels,
                                        embedding_labels = embedding_labels)
    >>> results.evaluate(category = ["global", "local", "downstream"])

    Evaluating global...
    Evaluating local...
    Evaluating downstream...
    /mnt/d/cytof/CytofDR/CytofDR/dr.py:263: UserWarning: No 'original_sell_types': Cell type-clustering concordance is not evaluated.
    warnings.warn("No 'original_sell_types': Cell type-clustering concordance is not evaluated.")
    /mnt/d/cytof/CytofDR/CytofDR/dr.py:263: UserWarning: No 'original_sell_types': Cell type-clustering concordance is not evaluated.
    warnings.warn("No 'original_sell_types': Cell type-clustering concordance is not evaluated.")
    /mnt/d/cytof/CytofDR/CytofDR/dr.py:263: UserWarning: No 'original_sell_types': Cell type-clustering concordance is not evaluated.
    warnings.warn("No 'original_sell_types': Cell type-clustering concordance is not evaluated.")

    >>> results.rank_dr_methods()

    {'PCA': 2.0416666666666665, 'ICA': 1.4583333333333333, 'UMAP': 2.5}

This runs successfully, but notice that a warning message has been generated! This is
okay because it is for informational purposes only. We still got our rankings and 
evaluations from this, depsite the slight change in averaged ranks.

.. note::

    In the case that Cell Type-Clustering Concordance is not performed, the ranking
    system is adjusted accordingly. So, the weights listed above is no longer
    used.


Concordance
-------------

For concordance, you will need a few more things: namely a comparison expression matrix, its cell types,
and the cell types of each embedding. The cell types are not necessarily easy to get. To demonstrate the
formats, we have:

.. code-block:: python

    >>> embedding_cell_types

    {'PCA': array(['CD4T', 'CD4T', 'unassigned', ..., 'unassigned', 'CD4T', 'CD4T'], dtype='<U10'),
     'UMAP': array(['CD8T', 'CD4T', 'unassigned', ..., 'unassigned', 'CD4T', 'CD4T'], dtype='<U11'),
     'ICA': array(['CD4T', 'CD4T', 'NK', ..., 'NK', 'CD4T', 'CD8T'], dtype='<U10')}


    >>> comparison_data

    array([[0.        , 0.        , 0.        , ..., 7.74983338, 6.49814686, 5.80650478],
           [0.        , 0.        , 5.45412597, ..., 8.22269542, 8.81728217, 6.83720621],
           [0.        , 0.        , 0.        , ..., 7.59089168, 8.28378631, 7.03165459],
           ...,
           [0.        , 0.        , 0.        , ..., 7.92024749, 7.66903686, 7.10977131],
           [0.        , 0.        , 0.        , ..., 7.77885408, 7.37359819, 6.68107842],
           [0.        , 0.        , 0.        , ..., 8.24210013, 8.61358196, 6.34647124]])

    >>> comparison_cell_types

    array(['CD4T', 'CD4T', 'CD4T', ..., 'CD4T', 'CD4T', 'Macrophages'], dtype='<U11')


And with these, we can modify our pipelines slightly to run concordance along with other methods:

.. code-block:: python

    >>> results.add_evaluation_metadata(original_data = expression,
    ...                                 original_labels = original_labels,
    ...                                 original_cell_types = original_cell_types,
    ...                                 embedding_labels = embedding_labels,
    ...                                 embedding_cell_types = embedding_cell_types,
    ...                                 comparison_data = comparison_data,
    ...                                 comparison_cell_types = comparison_cell_types)
    >>> results.evaluate(category = ["global", "local", "downstream", "concordance"])

    Evaluating global...
    Evaluating local...
    Evaluating downstream...
    Evaluating concordance...

    >>> results.rank_dr_methods()

    {'PCA': 1.8958333333333335, 'ICA': 1.4791666666666665, 'UMAP': 2.625}

And this is how you run concordance!