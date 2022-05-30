import numpy as np
import scipy
from sklearn.decomposition import FastICA, PCA, FactorAnalysis, KernelPCA, NMF
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, TSNE

import umap
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
import phate

import scipy.spatial
import scipy.stats
from CytofDR.metric import EvaluationMetrics, PointClusterDistance

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from typing import Union, Optional, List, Dict, Any

METHODS: Dict[str, bool]  = {"saucie": True, "ZIFA": True, "Grandprix": True}
    
try:
    import SAUCIE
except ImportError:
    METHODS["saucie"] = False
    print("No 'saucie' implementation.")
    
try:
    from ZIFA import ZIFA
except ImportError:
    METHODS["ZIFA"] = False
    print("No 'ZIFA' implementation.")
     
try:
    from GrandPrix import GrandPrix
except ImportError:
    METHODS["GrandPrix"] = False
    print("No 'Grandprix' implementation.")
    
    
def _verbose(message: str, verbose:bool=True):
    if verbose:
        print(message)
    
  
class Reductions():
    """A class for reductions and their evaluation. 

    This class is a convenient data class for storing and evaluaqting reductions.

    :param reductions: A dictionary of reductions as indexed by their names.
    
    :Attributes:
    - **original_data**: The original space data before DR.
    - **original_labels**: Clusterings based on original space data.
    - **original_cell_types**: Cell types based on original space data. 
    - **embedding_data**: The embedding space reduction.
    - **embedding_labels**: Clusterings based on embedding space reduction.
    - **embedding_cell_types**: Cell types based on embedding space reduction. 
    - **comparison_data**: The comparison data (matched with original data in some way) for concordance analysis.
    - **comparison_cell_types**: Cell types based on comparison data.
    - **comparison_classes**: Common cell types between embedding and comparison data.
    """
    
    def __init__(self, reductions: Optional[Dict[str, "np.ndarray"]]=None):
        """Constructor method for Reductions.
        """
        if reductions is None:
            self.reductions: Dict[str, "np.ndarray"] = {}
        else:
            self.reductions = reductions
            
        self.original_data: Optional["np.ndarray"] = None
        self.original_labels: Optional["np.ndarray"] = None
        self.original_cell_types: Optional["np.ndarray"] = None
        self.embedding_labels: Optional[Dict[str, "np.ndarray"]] = None
        self.embedding_cell_types: Optional[Dict[str, "np.ndarray"]] = None
        self.comparison_data: Optional["np.ndarray"] = None
        self.comparison_cell_types: Optional["np.ndarray"] = None
        self.comparison_classes: Optional[Union[str, List[str]]] = None
        
        self.evaluations: Dict[str, Any] = {}
    
    
    def add_reduction(self, reduction: "np.ndarray", name: str, replace: bool=False):
        """Add a reduction embedding.

        This method allows users to add additional embeddings.

        :param reduction: The reduction array.
        :param name: The name of the reduction.
        :param replace: If the original name exists, whether to replace the original, defaults to False
        
        :raises ValueError: Reduction already exists but users choose not to replace the original.
        """
        if name in list(self.reductions.keys()) and not replace:
            raise ValueError("Reduction already exists. Set 'replace' to True if replacement is intended.")
        self.reductions[name] = reduction
        
        
    def get_reduction(self, name: str):
        """Retrieve a reduction my name.

        This method allows users to retrieve a reduction by name. This equivalent to running ``self.reductions[name]``.

        :param name: The name of the reduction.
        """
        return self.reductions[name]
    
    
    def add_evaluation_metadata(self,
                                original_data: "np.ndarray",
                                original_labels: Optional["np.ndarray"]=None,
                                original_cell_types: Optional["np.ndarray"]=None,
                                embedding_labels: Optional[Dict[str, "np.ndarray"]]=None,
                                embedding_cell_types: Optional[Dict[str, "np.ndarray"]]=None,
                                comparison_data: Optional["np.ndarray"]=None,
                                comparison_cell_types: Optional["np.ndarray"]=None,
                                comparison_classes: Optional[Union[str, List[str]]]=None,
                                ):
        """Add supporting metadata for DR evaluation.

        A few more metadata are crucial for evaluation. ``original_data``, ``original_labels``,
        and ``embedding_labels`` are mendatory. Other metadata are optional, based on the metrics you
        wish to run.
        
        :param original_data: The original space data before DR.
        :param original_labels: Clusterings based on original space data.
        :param original_cell_types: Cell types based on original space data. 
        :param embedding_data: The embedding space reduction.
        :param embedding_labels: Clusterings based on embedding space reduction.
        :param embedding_cell_types: Cell types based on embedding space reduction. 
        :param comparison_data: The comparison data (matched with original data in some way) for concordance analysis.
        :param comparison_cell_types: Cell types based on comparison data.
        :param comparison_classes: Common cell types between embedding and comparison data.
        """
        self.original_data = original_data
        self.original_data = original_data
        self.original_labels = original_labels
        self.original_cell_types = original_cell_types
        self.embedding_labels = embedding_labels
        self.embedding_cell_types = embedding_cell_types
        self.comparison_data = comparison_data
        self.comparison_cell_types = comparison_cell_types
        self.comparison_classes = comparison_classes
        
    
    def evaluate(self,
                 category: Union[str, List[str]],
                 pwd_metric: str="PCD",
                 k_neighbors: int=5,
                 annoy_original_data_path: Optional[str]=None,
                 verbose: bool=True):     
        """Evaluate DR Methods Using Default DR Evaluation Scheme.

        This method ranks the DR methods based on any of the four default categories:
        ``global``, ``local``, ``downstream``, or ``concordance``. 
        
        :param category: The major evaluation category: ``global``, ``local``, ``downstream``, or ``concordance``.
        :param pwd_metric: The pairwise distance metric. Two options are "PCD" or "pairwise".
            PCD refers to Point Cluster Distance as implemented in this package; pairwise 
            is the traditional pairwise distance. For large datasets, PCD is recommended. Defaults to "PCD".
        :param k_neighbors: The number of neighbors to use for ``local`` metrics. Defaults to 5.
        :param annoy_original_data_path: The file path to an ANNOY object for original data.
        
        :raises ValueError: No reductions to evalate.
        :raises ValueError: Unsupported 'pwd_metric': 'PCD' or 'Pairwise' only.
        :raises ValueError: Evaluation needs 'original_data', 'original_labels', and 'embedding_labels' attributes.
        

        .. note::
        
            This method required ``add_evaluation_metadata`` to run first. ``original_cell_types`` and
            ``embedding_cell_types`` are optional for the downstream category. For ``concordance``, if
            you wish to use clustering results for embedding and comparison files, set the appropriate
            clusterings to ``embedding_cell_types`` and ``comparison_cell_types``. 
        """
        
        if len(self.reductions) == 0:
            raise ValueError("No reductions to evalate. Add your reductions first.")
        if pwd_metric.lower() not in  ["pcd", "pairwise"]:
            raise ValueError("Unsupported 'pwd_metric': 'PCD' or 'Pairwise' only.")
        
        self.evaluations = {}
                
        if isinstance(category, list):
            category = [c.lower() for c in category]
        else:
            category = [category.lower()]
                    
        if self.original_data is None or self.original_labels is None or self.embedding_labels is None:
            message: str = "Evaluation needs 'original_data', 'original_labels', and 'embedding_labels' attributes. "
            message += "Run 'add_evaluation_metadata()' methods first."
            raise ValueError(message)
        
        e: str
        if "global" in category:
            _verbose("Evaluating global...", verbose=verbose)
            self.evaluations["global"] = {"spearman": {}, "emd": {}}
            data_distance: "np.ndarray"
            embedding_distance: Dict[str, "np.ndarray"] = {}
            
            if pwd_metric.lower() == "pcd":
                data_distance = PointClusterDistance(self.original_data, self.original_labels).fit(flatten=True)
                for e in self.reductions.keys():
                    embedding_distance[e] = PointClusterDistance(self.reductions[e], self.original_labels).fit(flatten=True)
                
            else:
                data_distance = scipy.spatial.distance.pdist(self.original_data)
                for e in self.reductions.keys():
                    embedding_distance[e] = scipy.spatial.distance.pdist(e, metric="euclidean")
                    
            for e in self.reductions.keys():
                val: float = EvaluationMetrics.correlation(x=data_distance, y=embedding_distance[e], metric="Spearman")
                self.evaluations["global"]["spearman"][e] = val
                
                val: float = EvaluationMetrics.EMD(x=data_distance, y=embedding_distance[e])
                self.evaluations["global"]["emd"][e] = val
                   
        if "local" in category:
            _verbose("Evaluating local...", verbose=verbose)
            assert self.original_labels is not None
            self.evaluations["local"] = {"knn": {}, "npe": {}}
            
            data_neighbors: "np.ndarray" = EvaluationMetrics.build_annoy(self.original_data, annoy_original_data_path, k_neighbors)
            for e in self.reductions.keys():
                embedding_neighbors: "np.ndarray" = EvaluationMetrics.build_annoy(self.reductions[e], None, k_neighbors)
                self.evaluations["local"]["npe"][e] = EvaluationMetrics.NPE(labels = self.original_labels, data_neighbors=data_neighbors, embedding_neighbors=embedding_neighbors)
                self.evaluations["local"]["knn"][e] = EvaluationMetrics.KNN(data_neighbors=data_neighbors, embedding_neighbors=embedding_neighbors)
                             
        if "downstream" in category:
            _verbose("Evaluating downstream...", verbose=verbose)
            self.evaluations["downstream"] = {"cluster reconstruction: silhouette": {},
                                              "cluster reconstruction: DBI": {},
                                              "cluster reconstruction: CHI": {},
                                              "cluster reconstruction: RF": {},
                                              "cluster concordance: ARI": {},
                                              "cluster concordance: NMI": {},
                                              "cell type-clustering concordance: ARI": {},
                                              "cell type-clustering concordance: NMI": {}}
            for e in self.reductions.keys():
                self.evaluations["downstream"]["cluster reconstruction: silhouette"][e] = EvaluationMetrics.silhouette(embedding=self.reductions[e],
                                                                                                             labels=self.original_labels)
                self.evaluations["downstream"]["cluster reconstruction: DBI"][e] = EvaluationMetrics.davies_bouldin(embedding=self.reductions[e],
                                                                                                                    labels=self.original_labels)
                self.evaluations["downstream"]["cluster reconstruction: CHI"][e] = EvaluationMetrics.calinski_harabasz(embedding=self.reductions[e],
                                                                                                                       labels=self.original_labels)
                self.evaluations["downstream"]["cluster reconstruction: RF"][e] = EvaluationMetrics.random_forest(embedding=self.reductions[e],
                                                                                                                 labels=self.original_labels)
                
                self.evaluations["downstream"]["cluster concordance: ARI"][e] = EvaluationMetrics.ARI(x_labels=self.original_labels,
                                                                                                      y_labels=self.embedding_labels[e])
                self.evaluations["downstream"]["cluster concordance: NMI"][e] = EvaluationMetrics.NMI(x_labels=self.original_labels,
                                                                                                      y_labels=self.embedding_labels[e])
                
                if self.original_cell_types is not None:
                    self.evaluations["downstream"]["cell type-clustering concordance: ARI"][e] = EvaluationMetrics.ARI(x_labels=self.original_cell_types,
                                                                                                                    y_labels=self.embedding_labels[e])
                    self.evaluations["downstream"]["cell type-clustering concordance: NMI"][e] = EvaluationMetrics.NMI(x_labels=self.original_cell_types,
                                                                                                                    y_labels=self.embedding_labels[e])
                else:
                    warnings.warn("")
            
        if "condordance" in category:
            _verbose("Evaluating concordance...", verbose=verbose)
            assert self.embedding_cell_types is not None
            assert self.comparison_cell_types is not None
            assert self.comparison_data is not None
            self.evaluations["concordance"] = {"cluster distance": {}, "emd": {}, "gating concordance: ARI": {}, "gating concordance: NMI": {}}
            for e in self.reductions.keys():
                self.evaluations["concordance"]["emd"][e] = EvaluationMetrics.embedding_concordance(self.reductions[e],
                                                                                                    self.embedding_cell_types[e],
                                                                                                    self.comparison_data,
                                                                                                    self.comparison_cell_types,
                                                                                                    self.comparison_classes,
                                                                                                    "emd")
                self.evaluations["concordance"]["cluster distance"][e] = EvaluationMetrics.embedding_concordance(self.reductions[e],
                                                                                                                 self.embedding_cell_types[e],
                                                                                                                 self.comparison_data,
                                                                                                                 self.comparison_cell_types,
                                                                                                                 self.comparison_classes,
                                                                                                                 "cluster_distance")
                self.evaluations["concordance"]["gating concordance: ARI"][e] = EvaluationMetrics.ARI(x_labels=self.embedding_cell_types[e],
                                                                                                      y_labels=self.comparison_cell_types)
                self.evaluations["concordance"]["gating concordance: NMI"][e] = EvaluationMetrics.NMI(x_labels=self.embedding_cell_types[e],
                                                                                                      y_labels=self.comparison_cell_types)
          

    def rank_dr_methods(self, tie_method: str="max"):
        """Rank DR Methods Using Default DR Evaluation.

        Based on the results from the ``evaluate`` method, this method ranks the DR methods
        based on the categories chosen. All weighting schemes are consistent with the paper.
        Custom evaluation and weighting schemes are not supported in this case.
        
        :param tie_method: The method to deal with ties when ranking, defaults to "max".

        :return: A dictionary of DR methods and their final weighted ranks.
        """
        category_counter: int = 0
        overall_rank: np.ndarray = np.zeros(len(self.reductions))
        if "global" in self.evaluations.keys():
            category_counter += 1
            global_eval: np.ndarray = scipy.stats.rankdata(np.array(list(self.evaluations["global"]["spearman"].values())), method=tie_method)/2
            global_eval += scipy.stats.rankdata(-np.array(list(self.evaluations["global"]["emd"].values())), method=tie_method)/2
            overall_rank += global_eval
            
        if "local" in self.evaluations.keys():
            category_counter += 1
            local_eval: np.ndarray = scipy.stats.rankdata(np.array(list(self.evaluations["local"]["knn"].values())), method=tie_method)/2
            local_eval += scipy.stats.rankdata(-np.array(list(self.evaluations["local"]["npe"].values())), method=tie_method)/2
            overall_rank += local_eval
    
        if "downstream" in self.evaluations.keys():
            category_counter += 1
            cluster_reconstruction_eval: np.ndarray = scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cluster reconstruction: RF"].values())), method=tie_method)/4
            cluster_reconstruction_eval += scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cluster reconstruction: silhouette"].values())), method=tie_method)/4
            cluster_reconstruction_eval += scipy.stats.rankdata(-np.array(list(self.evaluations["downstream"]["cluster reconstruction: DBI"].values())), method=tie_method)/4
            cluster_reconstruction_eval += scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cluster reconstruction: CHI"].values())), method=tie_method)/4
            
            cluster_concordance_eval: np.ndarray = scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cluster concordance: ARI"].values())), method=tie_method)/2
            cluster_concordance_eval += scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cluster concordance: NMI"].values())), method=tie_method)/2
            
            if len(self.evaluations["downstream"]["cell type-clustering concordance: ARI"]) > 0:
                type_cluster_concordance_eval: np.ndarray = scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cell type-clustering concordance: ARI"].values())), method=tie_method)/2
                type_cluster_concordance_eval += scipy.stats.rankdata(np.array(list(self.evaluations["downstream"]["cell type-clustering concordance: ARI"].values())), method=tie_method)/2
                downstream_eval: np.ndarray = (cluster_reconstruction_eval + cluster_concordance_eval + type_cluster_concordance_eval)/3
            else:
                downstream_eval: np.ndarray = (cluster_reconstruction_eval + cluster_concordance_eval)/2
            overall_rank += downstream_eval
                
        if "concordance" in self.evaluations.keys():
            category_counter += 1
            concordance_eval: np.ndarray = scipy.stats.rankdata(-np.array(list(self.evaluations["concordance"]["cluster distance"].values())), method=tie_method)/3
            concordance_eval += scipy.stats.rankdata(-np.array(list(self.evaluations["concordance"]["emd"].values())), method=tie_method)/3
            concordance_eval += scipy.stats.rankdata(np.array(list(self.evaluations["concordance"]["gating concordance: ARI"].values())), method=tie_method)/6
            concordance_eval += scipy.stats.rankdata(np.array(list(self.evaluations["concordance"]["gating concordance: NMI"].values())), method=tie_method)/6
            overall_rank += concordance_eval
            
        overall_rank = overall_rank/category_counter
        return dict(zip(self.reductions.keys(), list(overall_rank)))
    
    
    def custom_evaluate(self):
        pass


    def rank_dr_method_custom(self):
        pass
                
            
    def plot_reduction(self, name: str, save_path: str, style: str="darkgrid", hue: Optional["np.ndarray"]=None, **kwargs):
        """Draw embedding using a scatter plot.

        This method generates a scatter plot for reductions in the class. 
        
        :param name: The name of the reduction.
        :param save_path: The path to save the plot.
        :param stype: The plot style, defaults to "darkgrid".
        :param hue: Labels used to color the points.
        :param kwargs: Keyword arguments passed into the ``sns.scatterplot`` method.
        
        .. note:: Live viewing is not supported by this method.
        """
        plt.figure()
        sns.set_style(style)
        sns_plot = sns.scatterplot(x=self.reductions[name][:,0], y=self.reductions[name][:,1], hue=hue, **kwargs)
        fig = sns_plot.get_figure()
        fig.savefig(save_path)
        plt.clf()
    
    
class LinearMethods():
    """Linear DR Methods
    
    This class contains static methods of a group of Linear DR Methods. If
    available, the sklearn implementation is used. All keyword arguments are
    passed directly to the method itself to allow for flexibility.
    """
    
    @staticmethod
    def PCA(data: "np.ndarray",
            out_dims: int=2,
            **kwargs
            ) -> "np.ndarray":
        """Scikit-Learn Principal Component Analysis (PCA)

        This method uses the Sklearn's standard PCA.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        return PCA(n_components=out_dims, **kwargs).fit_transform(data)
    
    
    @staticmethod
    def ICA(data: "np.ndarray",
            out_dims: int=2,
            **kwargs) -> "np.ndarray":   
        """Scikit-Learn Independent Component Analysis (ICA)

        This method uses the SKlearn's FastICA implementation of ICA.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        return FastICA(n_components=out_dims, **kwargs).fit_transform(data) #type: ignore
    
    
    @staticmethod
    def ZIFA(data: "np.ndarray",
             out_dims: int=2,
             **kwargs) -> "np.ndarray":
        """Zero-Inflated Factor Analysis (ZIFA)

        This method implements ZIFA as developed by Pierson & Yau (2015).
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        # Fix all-zero columns
        nonzero_col: List[int] = []
        col: int
        for col in range(data.shape[1]):
            if not np.all(data[:,col]==0):
                nonzero_col.append(col)
        data = data[:, nonzero_col]
        
        z: "np.ndarray"
        z, _ = ZIFA.fitModel(data, out_dims, **kwargs)
        
        return z
    
    
    @staticmethod
    def factor_analysis(data: "np.ndarray",
                        out_dims: int=2,
                        **kwargs) -> "np.ndarray":
        """Scikit-Learn Factor Analysis (FA)

        This method uses the SKlearn's FA implementation.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        return FactorAnalysis(n_components=out_dims, **kwargs).fit_transform(data)
    
    
    @staticmethod
    def NMF(data: "np.ndarray",
            out_dims: int=2,
            **kwargs) -> "np.ndarray":
        """Scikit-Learn Nonnegative Matrix Factorization (NMF)

        This method uses the SKlearn's NMF implementation.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        return NMF(n_components=out_dims, init = "nndsvd", **kwargs).fit_transform(data)
    
    
class NonLinearMethods():
    """NonLinear DR Methods.

    This class contains static methods of a group of NonLinear DR Methods, except for tSNE.
    """
    
    @staticmethod
    def MDS(data: "np.ndarray",
            out_dims: int=2,
            n_jobs: int=-1,
            **kwargs) -> "np.ndarray":
        """Scikit-Learn Multi-Dimensional Scaling (MDS)

        This method uses the SKlearn's MDS implementation.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return MDS(n_components=out_dims,
                   n_jobs=n_jobs,
                   **kwargs
                   ).fit_transform(data)
    
    
    @staticmethod
    def UMAP(data: "np.ndarray",
             out_dims: int=2,
             n_jobs: int=-1,
             **kwargs
             ) -> "np.ndarray":
        """UMAP

        This method uses the UMAP package's UMAP implementation.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return umap.UMAP(n_components=out_dims,
                         n_jobs=n_jobs,
                         **kwargs
                         ).fit_transform(data) #type: ignore
    
    
    @staticmethod
    def saucie(data: "np.ndarray",
               steps: int=1000,
               batch_size: int=256,
               **kwargs
               ) -> "np.ndarray":
        """SAUCIE

        This method is a wrapper for SAUCIE package's SAUCIE model. Specifically,
        dimension reduction is of interest. Here, all keyword arguments are passed
        into the ``SAUCIE.SAUCIE`` method. The training parameters ``steps`` and 
        ``batch_size`` are directly exposed in this wrapper.
        
        :param data: The input high-dimensional array.
        :param steps: The number of training steps to use, defaults to 1000.
        :param batch_size: The batch size for training, defaults to 256.

        :return: The low-dimensional embedding.
        """
        
        saucie: "SAUCIE.model.SAUCIE" = SAUCIE.SAUCIE(data.shape[1], **kwargs)
        train: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=True)
        saucie.train(train, steps=steps, batch_size=batch_size)
        
        eval: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=False)
        embedding: "np.ndarray" = saucie.get_embedding(eval) #type: ignore
        
        return embedding
    
    
    @staticmethod
    def isomap(data: "np.ndarray",
               out_dims: int=2,
               transform: Optional["np.ndarray"]=None,
               n_jobs: int=-1,
               **kwargs
               ) -> "np.ndarray":
        """Scikit-Learn Isomap

        This method is a wrapper for sklearn's implementation of Isomap.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param transform: The array to transform with the trained model.
        :param n_jobs: The number of threads to use.

        :return: The low-dimensional embedding.
        """
        
        if transform is None:
            embedding: "np.ndarray" = Isomap(n_components=out_dims,
                                             n_jobs=n_jobs,
                                             **kwargs).fit_transform(data)
        else:
            embedding: "np.ndarray" = Isomap(n_components=out_dims,
                                             n_jobs=n_jobs,
                                             **kwargs).fit(data).transform(transform)
        
        return embedding
    
    
    @staticmethod
    def LLE(data: "np.ndarray",
            out_dims: int=2,
            transform: Optional["np.ndarray"]=None,
            n_jobs: int=-1,
            **kwargs
            ) -> "np.ndarray": 
        """Scikit-Learn Locally Linear Embedding (LLE)

        This method is a wrapper for sklearn's implementation of LLE.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param transform: The array to transform with the trained model.

        :return: The low-dimensional embedding.
        """
        
        if transform is None:
            embedding: "np.ndarray" = LocallyLinearEmbedding(n_components=out_dims,
                                                             n_jobs=n_jobs,
                                                             **kwargs).fit_transform(data)
        else:
            embedding: "np.ndarray" = LocallyLinearEmbedding(n_components=out_dims,
                                                             n_jobs=n_jobs,
                                                             **kwargs).fit(data).transform(transform)
        
        return embedding
    
    
    @staticmethod
    def kernelPCA(data: "np.ndarray",
                  out_dims: int=2,
                  kernel: str="poly",
                  n_jobs: int=-1,
                  **kwargs
                  ) -> "np.ndarray": 
        """Scikit-Learn Kernel PCA

        This method is a wrapper for sklearn's implementation of kernel PCA.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param kernel: The kernel to use: "poly," "linear," "rbf," "sigmoid," or "cosine."
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return KernelPCA(n_components=out_dims,
                         kernel=kernel,
                         n_jobs=n_jobs,
                         **kwargs).fit_transform(data)


    @staticmethod
    def spectral(data: "np.ndarray",
                 out_dims: int=2,
                 n_jobs: int=-1,
                 **kwargs):
        """Scikit-Learn Spectral Embedding

        This method is a wrapper for sklearn's implementation of spectral embedding.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return SpectralEmbedding(n_components=out_dims,
                                 n_jobs=n_jobs,
                                 **kwargs).fit_transform(data)

    
    @staticmethod
    def phate(data: "np.ndarray",
              out_dims: int=2,
              n_jobs: int=-1,
              **kwargs):
        """PHATE

        This method is a wrapper for PHATE.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return phate.PHATE(n_components=out_dims,
                           n_jobs=n_jobs,
                           **kwargs).fit_transform(data)
    
    
    @staticmethod
    def grandprix(data: "np.ndarray",
                  out_dims: int=2,
                  **kwargs):
        """GrandPrix

        This method is a wrapper for GrandPrix.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.

        :return: The low-dimensional embedding.
        """
        return GrandPrix.fit_model(data = data, n_latent_dims = out_dims, **kwargs)[0]
    
    
    @staticmethod
    def sklearn_tsne(data: "np.ndarray",
                     out_dims: int=2,
                     n_jobs: int=-1,
                     **kwargs
                     )-> "np.ndarray":
        """Scikit-Learn t-SNE

        This method uses the Scikit-learn implementation of t-SNE. It supports both
        traditional and BH t-SNE with more control of variables.
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.

        :return: The low-dimensional embedding.
        """
        return TSNE(n_components=out_dims,
                    n_jobs=n_jobs,
                    **kwargs
                    ).fit_transform(data)
    
    
    @staticmethod
    def open_tsne(data: "np.ndarray",
                  out_dims: int=2,
                  perp: Union[List[int],int]=30,
                  learning_rate: Union[str, float]="auto",
                  early_exaggeration_iter: int=250,
                  early_exaggeration: float=12,
                  max_iter: int=500,
                  metric: str="euclidean",
                  dof: int=1,
                  theta: float=0.5, 
                  init: Union["np.ndarray", str]="pca",
                  negative_gradient_method: str="fft",
                  n_jobs: int=-1
                  ) -> "np.ndarray":
        """openTSNE implementation of FIt-SNE

        This is the Python implementation of FIt-SNE through the ``openTSNE`` package. Its implementation
        is based on research from Linderman et al. (2019). This is the default recommended implementation.
        To allow for flexibility and avoid confusion, common parameters are directly exposed without allowing
        additional keyword arguments. 
        
        :param data: The input high-dimensional array.
        :param out_dims: The number of dimensions of the output, defaults to 2.
        :param perp: Perplexity. The default is set to 30. Tradition is between 30 and 50.
            This also supports multiple perplexities with a list, defaults to 30.
        :param learning_rate: The learning rate used during gradient descent, defaults to "auto".
        :param early_exaggeration_iter: Number of early exaggeration iterations, defaults to 250.
        :param early_exaggeration: Early exaggeration factor, defaults to 12.
        :param max_iter: Maximum number of iterations to optimize, defaults to 500
        :param dof: T-distribution degree of freedom, defaults to "euclidean"
        :param theta: The speed/accuracy trade-off, defaults to 0.5.
        :param init: Method of initialiazation. 'random', 'pca', 'spectral', or array, defaults to "pca"
        :negative_gradient_method: Whether to use "bh" or "fft" tSNE, defaults to "fft".
        :param n_jobs: The number of jobs to run concurrantly, defaults to -1.
        
        :return: The low-dimensional embedding.
        """
        
        n_iter: int = max_iter - early_exaggeration_iter
        affinities_array: Union["affinity.PerplexityBasedNN", "affinity.Multiscale"]
        init_array: "np.ndarray" = np.empty((data.shape[0], out_dims))
        
        if isinstance(perp, list) and len(perp) > 1:
            affinities_array = affinity.Multiscale(data=data,
                                                   perplexities=perp,
                                                   metric=metric,
                                                   n_jobs=n_jobs,
                                                   verbose=True)
        else:
            perp = perp[0] if isinstance(perp, list) else perp
            affinities_array = affinity.PerplexityBasedNN(data=data,
                                                          perplexity=perp,
                                                          metric=metric,
                                                          n_jobs=n_jobs,
                                                          verbose=True)

        if isinstance(init, str):
            if init == "pca":
                init_array = initialization.pca(data, n_components=out_dims)
            elif init == "spectral":
                init_array = initialization.spectral(affinities_array.P, n_components=out_dims)
            else:
                init_array = initialization.random(data, n_components=out_dims)
        else:
            init_array = init
        
        embedding: "np.ndarray" = TSNEEmbedding(embedding=init_array,
                                                affinities=affinities_array,
                                                negative_gradient_method=negative_gradient_method,
                                                learning_rate=learning_rate,
                                                theta=theta,
                                                dof=dof,
                                                n_jobs=n_jobs,
                                                verbose=True)
        # Early exaggeration
        embedding.optimize(n_iter=early_exaggeration_iter,
                           inplace=True,
                           exaggeration=early_exaggeration,
                           momentum=0.5)
        # Regular GD
        embedding.optimize(n_iter=n_iter,
                           inplace=True,
                           momentum=0.8)
        
        return embedding    
    

def run_dr_methods(data: "np.ndarray",
                   methods: Union[str, List[str]]="all",
                   out_dims: int=2,
                   transform: Optional["np.ndarray"]=None,
                   n_jobs: int=-1,
                   verbose: bool=True,
                   suppress_error_msg: bool=False
                   ) -> "Reductions":
    """Run dimension reduction methods.

    This is a one-size-fits-all dispatcher that runs all supported methods in the module. It
    supports running multiple methods at the same time at the sacrifice of some more
    granular control of parameters. If you would like more customization, run each method
    indicidually instead.
    
    :param data: The input high-dimensional array.
    :param methods: DR methods to run (not case sensitive).
    :param out_dims: Output dimension of DR.
    :param transform: An array to transform after training on the traning set.
    :param n_jobs: The number of jobs to run when applicable, defaults to -1.
    :param verbose: Whether to print out progress, defaults to ``True``.
    :param suppress_error_msg: Whether to suppress error messages print outs, defaults to ``False``.

    :return: A Reductions object with all dimension reductions.
    """
    
    if not isinstance(methods, list):
        methods = [methods]
    methods = [each_method.lower() for each_method in methods]
    
    if "all" in methods:
        methods = ["pca", "ica", "umap", "sklearn_tsne", "open_tsne", "factor_analysis", "isomap", "mds", "lle",
                   "kernelpca_poly", "kernelpca_rbf", "phate", "nmf", "spectral"]
        if METHODS["saucie"]:
            methods.append("saucie")
        if METHODS["zifa"]:
            methods.append("zifa")
        if METHODS["GrandPrix"]:
            methods.append("grandprix")
            
    reductions: Reductions = Reductions()
    
    if "pca" in methods:
        try:
            _verbose("Running pca", verbose=verbose)
            reductions.add_reduction(LinearMethods.PCA(data, out_dims=out_dims), "pca")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "ica" in methods:
        try:
            _verbose("Running ica", verbose=verbose)
            reductions.add_reduction(LinearMethods.ICA(data, out_dims=out_dims), "ica") 
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
        
    if "umap" in methods:
        try:
            _verbose("Running umap", verbose=verbose)    
            reductions.add_reduction(NonLinearMethods.UMAP(data, out_dims=out_dims, n_jobs=n_jobs), "umap")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "saucie" in methods:
        try:
            _verbose("Running saucie", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.saucie(data), "saucie")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
    
    # sklearn BH
    if "sklearn_tsne" in methods:
        try:
            _verbose("Running sklearn_tsne", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.sklearn_tsne(data, out_dims=out_dims, n_jobs=n_jobs), "sklearn_tsne")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
    
    if "open_tsne" in methods:
        try:
            _verbose("Running open_tsne", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.open_tsne(data, out_dims=out_dims, n_jobs=n_jobs), "open_tsne")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "zifa" in methods:
        try:
            _verbose("Running zifa", verbose=verbose)
            reductions.add_reduction(LinearMethods.ZIFA(data, out_dims=out_dims), "zifa")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "fa" in methods:
        try:
            _verbose("Running fa", verbose=verbose)
            reductions.add_reduction(LinearMethods.factor_analysis(data, out_dims=out_dims), "fa")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "isomap" in methods:
        try:
            _verbose("Running isomap", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.isomap(data, out_dims=out_dims,transform=transform, n_jobs=n_jobs), "isomap")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "mds" in methods:
        try:
            _verbose("Running mds", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.MDS(data, out_dims=out_dims, n_jobs=n_jobs), "mds")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "lle" in methods:
        try:
            _verbose("Running lle", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.LLE(data, out_dims=out_dims, transform=transform, n_jobs=n_jobs), "lle")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "spectral" in methods:
        try:
            _verbose("Running spectral", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.spectral(data, out_dims=out_dims, n_jobs=n_jobs), "spectral")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
                    
    if "kpca_poly" in methods:
        try:
            _verbose("Running kpca_poly", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.kernelPCA(data, out_dims=out_dims, kernel="poly", n_jobs=n_jobs), "kpca_poly")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "kpca_rbf" in methods:
        try:
            _verbose("Running kpca_rbf", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.kernelPCA(data, out_dims=out_dims, kernel="rbf", n_jobs=n_jobs), "kpca_rbf")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "phate" in methods:
        try:
            _verbose("Running phate", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.phate(data, out_dims=out_dims, n_jobs=n_jobs), "phate")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "nmf" in methods:
        try:
            _verbose("Running nmf", verbose=verbose)
            reductions.add_reduction(LinearMethods.NMF(data, out_dims=out_dims), "nmf")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    if "grandprix" in methods:
        try:
            _verbose("Running grandprix", verbose=verbose)
            reductions.add_reduction(NonLinearMethods.grandprix(data, out_dims=out_dims), "grandprix")
        except Exception as e:
            _verbose(str(e), verbose=not suppress_error_msg)
            
    return reductions
