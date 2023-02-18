import setuptools
import sys
import os
import shutil
import distutils.cmd

VERSION = "0.2.1"

class PypiCommand(distutils.cmd.Command):
    
    description = "Build and upload for PyPI."
    user_options = []
    
    def initialize_options(self):
        pass
    
    
    def finalize_options(self):
        pass
    
    
    def run(self):
        shutil.rmtree("dist/")
        
        wheel_file = "CytofDR-{}-py3-none-any.whl".format(VERSION)
        tar_file = "CytofDR-{}.tar.gz".format(VERSION)
        
        os.system("{} setup.py sdist bdist_wheel".format(sys.executable))
        os.system("twine upload dist/{} dist/{}".format(wheel_file, tar_file))
    
    
class CondaCommand(distutils.cmd.Command):
    
    description = "Build and upload for conda."
    user_options = []
    
    def initialize_options(self):
        pass
    
    
    def finalize_options(self):
        pass
    
    
    def run(self):
        shutil.rmtree("dist_conda/")
        os.system("conda build . --output-folder dist_conda/ -c conda-forge -c bioconda")
        os.system("anaconda upload ./dist_conda/noarch/cytofdr-{}-py_0.tar.bz2".format(VERSION))


setuptools.setup(
    name = "CytofDR",
    version = VERSION,
    description = "CyTOF Dimension Reduction Framework",
    author="Kevin Wang",
    url="https://github.com/kevin931/CytofDR",
    long_description_content_type = "text/markdown",
    long_description = open("README.md").read(),
    packages=["CytofDR"],
    python_requires=">=3.7",
    install_requires=["scikit-learn",
                      "numpy",
                      "scipy",
                      "umap-learn",
                      "openTSNE",
                      "phate",
                      "annoy",
                      "matplotlib",
                      "seaborn"],
    test_requires=["pytest",
                   "pytest-cov",
                   "pytest-mock",
                   "coverage"],
    license="MIT",
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English"
    ],
    cmdclass = {"pypi": PypiCommand,
                "conda": CondaCommand
                }
)