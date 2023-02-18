##########
Releases
##########

Here we include our release notes for past releases in sequential order.

--------------------

********
v0.3.0
********

This releases adds some new features along with some minor improvements and fixes.


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

********
v0.2.1
********

This is a minor maintenance update of v0.2.x with a few improvements on documentation and docstrings.

Changes and New Features
--------------------------

- Update licensing information

Improvements
---------------

- Improve documentation and docstrings

Deprecations
----------------

- (Since v0.2.0) The `comparison_classes` parameter of the `EvaluationMetrics.embedding_concordance` method will no longer accept `str` input.


********
v0.2.0
********

This releases adds some new features along with some minor improvements and fixes.


Changes and New Features
--------------------------

- Add `pairwise_downsample` option for pairwise distance optimization in `Reductions.evaluate` method.
- Add example datasets for the GitHub repo.
- Improve docstrings for documentations.
- Improve unit tests for coverage.

Deprecations
----------------

- The `comparison_classes` parameter of the `EvaluationMetrics.embedding_concordance` method will no longer accept `str` input.

*************************************
v0.1.1 (EOL)
*************************************

This is a minor update of v0.1.x with a few improvements on documentation and docstrings. Given the
release of v0.3.0, this version is the End-of-Life (EOL) release for v0.1.x. This version will no
longer be maintained going forward. For new features and future fixes, migrate to v0.2.0 or higher,
which is compatible with v0.1.x.

Changes and New Features
--------------------------

- Update licensing information

Improvements
---------------

- Improve documentation and docstrings


********
v0.1.0
********

This is the first official release of ``CytofDR`` with LTS.


Changes and New Features
--------------------------

- Support for magic methods: ``print`` and ``[]`` for ``Reductions`` class
- Add ``names`` attributes to ``Reductions`` class
- Add custom DR evaluation
- Add functions to save DR embeddings and evaluations
- Improve documentation and docstrings



********
v0.0.1
********

- This is the first offical pre-release of ``CytofDR``.
- Most of the pipeline is complete, including DR, evaluation, ranking, and plotting.
- Extensive documentation and tutorial complete.
- This release aims to aid the completion of our development and tool chain.
- We are on  ``conda`` and ``PyPI``!

.. note:: This is not an official stable release. Please wait for v0.1.0 in the near future.