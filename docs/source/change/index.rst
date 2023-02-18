#############
Changelog
#############

Here are the most recent releases and changes of cytomulate. Currently, we're still under developmet.
Therefore, we don't have any official releases. However, check out our git history to see what we're
doing!

------------------------

****************************
Latest Release: v0.3.0
****************************

This is a minor maintenance update of v0.2.x with a few improvements on documentation and docstrings.

Changes and New Features
--------------------------

- Add `minmax` normalization option for the `evaluation.EvaluationMetric.EMD` method
- Add alternative implementation of `evaluation.EvaluationMetric.NPE` with total variation distance (TVD)
- Allow both min-max EMD and and TVD NPE for automatic evaluation of DR methods

Improvements
--------------

- Docstrings and documentations reformatted for clarity

Deprecations
----------------

- (Since v0.2.0) The `comparison_classes` parameter of the `EvaluationMetrics.embedding_concordance` method will no longer accept `str` input.


.. toctree::
    :maxdepth: 1

    releases
    recent
