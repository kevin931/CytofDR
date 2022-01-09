import setuptools

VERSION = "0.0.0"

setuptools.setup(
    name = "CytofDR",
    version = VERSION,
    description = "Dimension Reduction Methods and Benchmarking Tools for CyTOF",
    packages=["CytofDR"],
    python_requires=">=3.9",
    install_requires=["scikit-learn",
                      "numpy",
                      "scipy",
                      "umap-learn",
                      "openTSNE",
                      "phate"],
    test_requires=["pytest",
                   "pytest-cov",
                   "pytest-mock",
                   "coverage"],
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English"
    ]
)