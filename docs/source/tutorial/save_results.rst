#################################
Save DR and Evaluation Results
#################################

After running DR and evaluating them, one of the things that you should consider
is to save them for later use. Admittedly, these results are often only an
intermediate step in your research or workflow, and also, some of these computations
are quite inefficient to rerun every time. Thus, this short tutorial shows you
how to save your results to dist for both your reductions and your evaluations.

-------------------------------

*******************
Save Reductions
*******************

There are two ways to save your embeddings. You can either you the hands-off
approach and save all embeddings at once, or you can save them yourself. The
former is really easy (as shown in the
`Quickstart Guide <https://cytofdr.readthedocs.io/en/latest/quickstart.html>`_):

.. code-block:: python

    >>> results.save_all_reductions(save_dir="YOUR_DIR_PATH", delimiter=",")

For the purpose of this tutorials, ``results`` is a ``Reductions`` object with
existing reductions and evaluations. This quick method will save all your
reductions with their embedding names as the file names. However, you may wish to
save them on your own for reasons such as needing only a few of the embeddings
after ranking the DR methods or wanting to name DR in your own ways, you can do
the following:

.. code-block:: python

    >>> results.save_reduction(name="umap", path="./my_umap.txt", delimiter=",")

This saves your ``UMAP`` embedding to your specified path with a different name.

There are a couple of other customizations you can do. One is the delimiter. If you
prefer a tsv or other deliminters, simply change that parameter will help you:

.. code-block:: python

    >>> results.save_all_reductions(save_dir="YOUR_DIR_PATH", delimiter="\t")

Further, if a file already exists in the path specified, we don't overwrite files
by default. Rather, we throw an ``FileExistsError`` to minimize any mistakes. However,
if you do wish to overwrite old files, you can specify the ``overwrite`` parameter:

.. code-block:: python

        >>> results.save_all_reductions(save_dir="YOUR_DIR_PATH", overwrite=True, delimiter="\t")

Overwriting will be handy sometimes without having to specifically delete files,
but it's not the default behavior, which can obviously cause issues.

-------------------------

****************************
Save Evaluations
****************************

After evaluating your DR, it's also possible to save your results for future
analyses. The interface is quite similar to save embeddings, except that you
can currently save it to a ``csv`` file:

.. code-block:: python

    >>> results.save_evaluations(path="YOUR_FILE_PATH")

Here, you do need to give it a full path. The resulting csv file will result
in a table that looks like this:

========== =========== ======== ======================
Category    Metric      Method   Value
---------- ----------- -------- ----------------------
global      spearman     PCA      0.37249992312498076
global      spearman     ICA      0.2901695885423971
global      emd          PCA      0.41181293853758305
global      emd          ICA      0.7883300728571819
local       knn          PCA      0.126
local       knn          ICA      0.126
local       npe          PCA      0.76
local       npe          ICA      0.76
========== =========== ======== ======================

You can easily run any downstream analyses using ``pandas`` or even ``R``
if you prefer.