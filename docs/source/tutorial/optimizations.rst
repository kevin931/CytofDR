Optimizations: PCD and ANNOY
==============================

With CyTOF's uniquely large sample sizes, we often struggle with memory and speed. These
issues are not trivial to address: people dedicate their careers to them. While we try
to make things fast, a lot of things are beyond us. What we can do is to choose algorithms
wisely and think hard when we have to implement our own. In this package, we use two
non-standard algorithms to speed things up: ``ANNOY`` and ``PointClusterDistance``. The
former is a famous nearest-neighbor algorithm that is off-the-shelf, whereas we introduced
and implemented the latter ourselves.

This page details how you can use these two optimizations, which come as defaults, in
our package.

-----------------------------

*********
ANNOY
*********

Do you find slow KNN algorithms annoying? (Get it? Okay, I will stop.) But seriously,
who has time to wait for neighbors? Thus, we were exhilarated when we found out about
`Approximated Nearest Neighbors Oh Yeah (ANNOY) <https://github.com/spotify/annoy>`_!!
This indeed is a popular algorithm, which is also used by packages such as ``openTSNE``,
and from our experiences, it is very (I mean very) fast! Contructing a neighbor graph
often takes within a minute even with large sample sizes. 

While the exact inner working of this algorithm is quite beyond us, we have included
an interface to python's ``annoy`` package in our ``evaluation`` module so that
you can easily use it. In fact, you likely won't need to use this API unless you
are planning on running evaluations yourself without our ``Reductions`` class. But
in case you need, you can use our ``Annoy`` class.

Building Annoy
---------------

We make it a breeze to do so! Here is an example with a random numpy array:

.. code-block:: python

    >>> from CytofDR.evaluation import Annoy
    >>> import numpy as np

    >>> expression = np.random.rand(100, 2)
    >>> Annoy.build_annoy(data = expression)
    <annoy.Annoy object at 0x7fa14159ceb0>

This is it! It's pretty easy, and you have an ``AnnoyIndex`` object to work with.
To find out how exactly it works, we recommend you look at their documentation!
However, if you want to have a neighbor array, we have another interface in
the ``EvaluationMetrics`` class. Here, you lose have the ability to specify
the number of neighbors:

.. code-block:: python

    >>> from CytofDR.evaluation import EvaluationMetrics
    >>> import numpy as np

    >>> expression = np.random.rand(100, 2)
    >>> EvaluationMetrics.build_annoy(data = expression, k = 10)
    array([[ 9, 21,  5, 78, 15, 46, 63,  3, 81, 26],
           [58, 87, 95, 97, 71, 53, 22, 68,  2,  8],
           [ 8, 68, 86, 11, 32, 22, 53, 71, 66, 52],
           [81, 26,  6, 46, 70, 15, 78, 30, 21,  0],
           [30, 70,  6, 26, 19, 81,  3, 40, 46, 15],
           ... ,
           [84, 80, 75, 13, 56, 23, 93, 76, 82, 59],
           [71, 53, 22, 68,  2,  8, 87,  1, 86, 11],
           [16, 67, 24, 44, 14, 51, 36, 74, 27, 62],
           [91, 12, 18, 17,  7, 34, 20, 10, 94, 85]])

``EvaluationMetrics.build_annoy()`` is a convenient wrapper for the ``Annoy``
class. And we recommend using that unless you need customizations.

Load and Save Annoy Objects
----------------------------

Sometimes, you want to save your Annoy object if you are using
the same dataset repeatedly and don't want to reconstruct it
every time. The ``Annoy`` library makes it easy to do so, and
in that spirit, we provide a convenient as well. To save it, 
you can simply do this:

.. code-block:: python

    >>> from CytofDR.evaluation import Annoy

    >>> annoy_object = Annoy.build_annoy(data = expression)
    >>> Annoy.save_annoy(annoy_object, path = "YOUR_PATH")


and it can be loaded easily:

.. code-block:: python

    >>> annoy_object = Annoy.load_annoy(path = "YOUR_PATH")

If you want to use the ``EvaluationMetrics`` interface, you can
pass in the path and it will automatically load it for you:

.. code-block:: python

    >>> EvaluationMetrics.build_annoy(data = expression, saved_annoy_path = "YOUR_PATH", k = 10)
    array([[ 9, 21,  5, 78, 15, 46, 63,  3, 81, 26],
           [58, 87, 95, 97, 71, 53, 22, 68,  2,  8],
           [ 8, 68, 86, 11, 32, 22, 53, 71, 66, 52],
           [81, 26,  6, 46, 70, 15, 78, 30, 21,  0],
           [30, 70,  6, 26, 19, 81,  3, 40, 46, 15],
           ... ,
           [84, 80, 75, 13, 56, 23, 93, 76, 82, 59],
           [71, 53, 22, 68,  2,  8, 87,  1, 86, 11],
           [16, 67, 24, 44, 14, 51, 36, 74, 27, 62],
           [91, 12, 18, 17,  7, 34, 20, 10, 94, 85]])

It's as easy as that!

-----------------------------

*****************************
Point Cluster Distance (PCD)
*****************************

Some of the evaluation metrics need to use a pairwise distance matrix. 
This sounds very simple: you load ``scipy.dist`` and you're on your
way. But wait! There is bad news: if your sample is large, I mean
larger than hundreds of thousands of cells, you will quickly run into
trouble because the memory complexity is O(N^2). This is scary because
it quickly outpaces what is reseaonable for any user.

There are a few solutions:

1. We find an alternative altogether without finding pairwise distance.
2. We compromise and find partial pairwise distance.

Here, PCD belongs to the second solution. Instead of finding the pairwise
between every cell, we utilze clusters! **The idea** is that with clusters,
we can easily compute cluster centroids, and by definition, we expect the
number of the clusters k to be much much smaller than the number of cell.
Thus, we compute the pairwise distance between cluster centroids and all
cells, effectively reducing the memory complexity to O(k*N). This makes
it feasible to work with dataset even in the realm of millions as long
as the cluster numbers and the feature numbers of the original expression
matrix is resaonble. While this sounds very easy and you can probably
implement it in 10 minutes, the good news is that we have already
implemented this (save the 10 minutes for a walk outside)! Now,
let's look at the interface!


Using the ``PointClusterDistance`` class
-----------------------------------------

The ``PointClusterDistance``'s interface is akin to that of the ``sklearn``'s.
Here, we need both the expression matrix and the labels. We will simulate them
using random arrays:

.. code-block:: python

    >>> import numpy as np

    >>> expression = np.random.rand(100, 2)
    array([[3.89241108e-01, 3.74894390e-01],
           [2.53484808e-02, 8.38034055e-01],
           [1.81716130e-01, 6.77570720e-01],
           [2.81618520e-01, 2.85328034e-01],
           [8.72380359e-01, 9.43282666e-01],
           ...,
           [5.64692819e-01, 3.01563659e-01],
           [1.52626514e-01, 9.19922564e-01],
           [9.13887763e-01, 1.12411075e-01],
           [8.86792310e-01, 2.78847601e-01]])

    >>> labels = np.repeat(["a", "b"], 50)
    array(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b',
           'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
           'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
           'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
           'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'], dtype='<U1')

Now, we can instantatiate our ``PointClusterDistance``:

.. code-block:: python

    >>> from CytofDR.evaluation import PointClusterDistance
    >>> pcd = PointClusterDistance(X = expression, labels = labels)
    >>> pcd.fit()
    array([[0.13338769, 0.11425074],
           [0.57807929, 0.57630871],
           [0.35672805, 0.35369442],
           [0.27244763, 0.25324668],
           ...
           [0.19573966, 0.19065184],
           [0.54919051, 0.55249102],
           [0.56914471, 0.57123187],
           [0.45559932, 0.4623146 ]])

This should look familiar to you! And the ``fit()`` method returns
a distance array by default: the rows are cells and the columns are
clusters. To get the indices for cluster labels

.. code-block:: python

    >>> pcd.index
    array(['a', 'b'], dtype='<U1')

which will allow you to construct what each distance represents. 

For convenience, you can also flatten the array into a 1-d array,
or simply the vector, so that you can run other metrics:

.. code-block:: python

    >>> pcd.fit(flatten = True)
    array([0.13338769, 0.11425074, 0.57807929, 0.57630871, 0.35672805,
       0.35369442, 0.27244763, 0.25324668, 0.6125959 , 0.63173534,
       ..., 0.55249102, 0.56914471, 0.57123187, 0.45559932, 0.4623146 ])
