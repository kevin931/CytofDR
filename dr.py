import numpy as np
import sklearn
from sklearn.decomposition import FastICA, PCA, FactorAnalysis, KernelPCA, NMF
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding

import umap
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
import phate

from fileio import FileIO

import time
import os
from typing import Tuple, Union, Optional, List, Dict

METHODS: Dict[str, bool]  = {"fit_sne": True, "bh_tsne": True, "saucie": True, "ZIFA": True}

try:
    from fitsne.fast_tsne import fast_tsne
except:
    METHODS["fit_sne"] = False
    print("No 'fit-sne' implementation. Please use 'open_tsne' instead.")

try:
    from bhtsne.bhtsne import run_bh_tsne
except:
    METHODS["bh_tsne"] = False
    print("No 'bh_tsne' implementation. Use 'sklearn_tsne_bh' or 'open_tsne' instead.")
    
try:
    import SAUCIE
except:
    METHODS["saucie"] = False
    print("No 'saucie' implementation.")
    
try:
    from ZIFA import ZIFA
except:
    METHODS["ZIFA"] = False
    print("No 'ZIFA' implementation.")
     
try:
    from GrandPrix import GrandPrix
except:
    METHODS["GrandPrix"] = False
    print("No 'Grandprix' implementation.")
    

class DR():
    """Dimension Reduction Class.

    This class is simply a wrapper for a utility function that dispatches
    various DR methods.
    """

    @classmethod
    def run_methods(cls,
                    data: "np.ndarray",
                    out: str,
                    transform: Optional["np.ndarray"]=None,
                    methods: Union[str, List[str]]="all",
                    out_dims: int=2,
                    save_embedding_colnames: bool=False,
                    perp: Union[int, List[int]]=30,
                    early_exaggeration: float=12.0,
                    early_exaggeration_iter: int=250,
                    tsne_learning_rate: Optional[Union[float, str]]=None,
                    max_iter: int=1000,
                    init: str="random",
                    dist_metric: str="euclidean",
                    open_tsne_method: str="fft",
                    umap_min_dist: float=0.1,
                    umap_neighbors: int=15,
                    SAUCIE_lambda_c: float=0.0,
                    SAUCIE_lambda_d: float=0.0,
                    SAUCIE_steps: int=1000,
                    SAUCIE_batch_size: int=256,
                    SAUCIE_learning_rate: float=0.001,
                    kernel: str="poly",
                    phate_decay: int=40,
                    phate_knn: int=5
                    ) -> List[List[Union[str, float]]]:
        
        ''' Run dimension reduction methods.
        
        This is a one-size-fits-all dispatcher that runs all supported methods in the module. It
        supports running multiple methods at the same time at the sacrifice of some more
        granular control of parameters. This method is also what the CLI uses.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out (str): The output directory to save embeddings.
            transform (numpy.ndarray): An array to transform after training on the traning set.
            methods (str, List[str]): DR methods to run (not case sensitive).
            out_dims (int): Output dimension of DR.
            save_embedding_colnames (bool): Whether to save the columnnames as the first row.
            perp (int, List[int]): Perplexity or multiple perplexity for tSNE.
            early_exaggeration (float): Early exaggeration for tSNE.
            early_exaggeration_iter (int): Early exaggeration iteration for tSNE.
            tsne_learning_rate (str, float, optional): Learning rate for tSNE.
            max_iter (int): Maximum number of iterations for tSNE.
            init (str): tSNE and UMAP initialization method.
            dist_metric (str): Distance metric for methods that support it.
            open_tsne_method (str): Method for openTSNE.
            umap_min_dist (float): Minimum distane for UMAP
            umap_neighbors (int): Number of neighborsi for UMAP.
            SAUCIE_lambda_c (float): ID regularization.
            SAUCIE_lambda_d (float): Within-cluster distance regularization.
            SAUCIE_steps (int): Number of iterations for SAUCIE
            SAUCIE_batch_size (int): Batch size for SAUCIE
            SAUCIE_learning_rate (float): Learning rate for SAUCIE
            kernel (str): Kernel for KernelPCA.
            phate_decay (int): Decay rate for PHATE.
            phate_knn (int): Number of neighbors in PHATE.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        '''
        
        dir_path: str = out+"/embedding"
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass
        
        if not isinstance(methods, list):
            methods = [methods]
        methods = [each_method.lower() for each_method in methods]
        
        if "all" in methods:
            methods = ["pca", "umap", "sklearn_tsne_original", "sklearn_tsne_bh", "open_tsne", "factor_analysis", "isomap", "mds"]
            if METHODS["fit_sne"]:
                methods.append("fit_sne")
            if METHODS["bh_tsne"]:
                methods.append("bh_tsne")
            if METHODS["saucie"]:
                methods.append("saucie")
            if METHODS["zifa"]:
                methods.append("zifa")
                
        if not isinstance(perp, list):
            perp = [perp]
            
        colnames: Optional[np.ndarray] = None
        if save_embedding_colnames:
            colnames = np.arange(out_dims)

        time: List[List[Union[str, float]]] = [[], []]
        time_PCA: float
        time_UMAP: float
        time_tsne_original: float
        time_sklearn_tsne_bh: float
        time_bh_tsne: float
        time_fit_sne: float
        time_open_tsne: float
        
        embedding_PCA: "np.ndarray"
        embedding_UMAP: "np.ndarray"
        embedding_tsne_original: "np.ndarray"
        embedding_sklearn_tsne_bh: "np.ndarray"
        embedding_bh_tsne: "np.ndarray"
        embedding_fit_sne: "np.ndarray"
        embedding_open_tsne: "np.ndarray"
        
        if tsne_learning_rate == "auto":
            tsne_learning_rate = round(data.shape[0]/12) if data.shape[0]>=2400 else 200.0
        elif tsne_learning_rate is not None:
            tsne_learning_rate = float(tsne_learning_rate)
        else:
            tsne_learning_rate = 200.0
        
        if "pca" in methods:
            try:
                time_PCA, embedding_PCA = LinearMethods.PCA(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_PCA, dir_path, "PCA", col_names=colnames)
                time[0].append("PCA")
                time[1].append(time_PCA)
            except Exception as e:
                print(e)
                
        if "ica" in methods:
            try:
                time_ICA, embedding_ICA = LinearMethods.ICA(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_ICA, dir_path, "ICA", col_names=colnames)
                time[0].append("ICA")
                time[1].append(time_ICA)
            except Exception as e:
                print(e)
            
        if "umap" in methods:
            try: 
                if init != "spectral" or init != "random":
                    init_umap: str = "spectral"
                else:
                    init_umap: str = init
                    
                time_UMAP, embedding_UMAP = NonLinearMethods.UMAP(data,
                                                                  out_dims=out_dims,
                                                                  init=init_umap,
                                                                  metric=dist_metric,
                                                                  min_dist=umap_min_dist,
                                                                  n_neighbors=umap_neighbors)
                FileIO.save_np_array(embedding_UMAP, dir_path, "UMAP", col_names=colnames)
                time[0].append("UMAP")
                time[1].append(time_UMAP)
            except Exception as e:
                print(e)
                
        if "saucie" in methods:
            try:
                time_saucie, embedding_saucie = NonLinearMethods.saucie(data,
                                                                        lambda_c=SAUCIE_lambda_c,
                                                                        lambda_d=SAUCIE_lambda_d,
                                                                        steps=SAUCIE_steps,
                                                                        batch_size=SAUCIE_batch_size,
                                                                        learning_rate=SAUCIE_learning_rate)
                FileIO.save_np_array(embedding_saucie, dir_path, "SAUCIE", col_names=colnames)
                time[0].append("SAUCIE")
                time[1].append(time_saucie)
            except Exception as e:
                print(e)
        
        # Sklearn original
        if "sklearn_tsne_original" in methods:
            try:
                time_tsne_original, embedding_tsne_original = TSNE.sklearn_tsne(data,
                                                                                out_dims=out_dims,
                                                                                perp=perp[0],
                                                                                early_exaggeration=early_exaggeration,
                                                                                learning_rate=tsne_learning_rate,
                                                                                max_iter=max_iter,
                                                                                method="exact",
                                                                                init=init,
                                                                                metric=dist_metric)
                FileIO.save_np_array(embedding_tsne_original, dir_path, "tsne_original", col_names=colnames)
                time[0].append("sklearn_tsne_original")
                time[1].append(time_tsne_original)
            except Exception as e:
                print(e)
        
        # sklearn BH
        if "sklearn_tsne_bh" in methods:
            try:
                time_sklearn_tsne_bh, embedding_sklearn_tsne_bh = TSNE.sklearn_tsne(data,
                                                                                    out_dims=out_dims,
                                                                                    perp=perp[0],
                                                                                    early_exaggeration=early_exaggeration,
                                                                                    learning_rate=tsne_learning_rate,
                                                                                    max_iter=max_iter,
                                                                                    init=init,
                                                                                    metric=dist_metric)
                FileIO.save_np_array(embedding_sklearn_tsne_bh, dir_path, "sklearn_tsne_bh", col_names=colnames)
                time[0].append("sklearn_tsne_bh")
                time[1].append(time_sklearn_tsne_bh)
            except Exception as e:
                print(e)
        
        if "bh_tsne" in methods:
            try:
                time_bh_tsne, embedding_bh_tsne = TSNE.bh_tsne(data,
                                                               out_dims=out_dims,
                                                               perp=perp[0],
                                                               max_iter=max_iter)
                FileIO.save_np_array(embedding_bh_tsne, dir_path, "bh_tsne", col_names=colnames)
                time[0].append("bh_tsne")
                time[1].append(time_bh_tsne)
            except Exception as e:
                print(e)
        
        if "fit_sne" in methods:
            perp_list: Optional[List[int]] = None
            perp_fit_sne: int = 0
            
            if len(perp)==1:
                perp_fit_sne = perp[0]
            else:
                perp_list = perp
                
            try:
                time_fit_sne, embedding_fit_sne = TSNE.fit_sne(data,
                                                               out_dims=out_dims,
                                                               perp=perp_fit_sne,
                                                               early_exaggeration=early_exaggeration,
                                                               stop_early_exag_iter=early_exaggeration_iter,
                                                               learning_rate=tsne_learning_rate,
                                                               max_iter=max_iter,
                                                               perplexity_list=perp_list, #type:ignore
                                                               init=init) 
                FileIO.save_np_array(embedding_fit_sne, dir_path, "fit_sne", col_names=colnames)
                time[0].append("fit_sne")
                time[1].append(time_fit_sne)
            except Exception as e:
                print(e)
                
            try:
                os.remove("data.dat")
                os.remove("result.dat")
            except Exception as e:
                print(e)
        
        if "open_tsne" in methods:
            try:
                time_open_tsne, embedding_open_tsne = TSNE.open_tsne(data,
                                                                     out_dims=out_dims,
                                                                     perp=perp,
                                                                     early_exaggeration=early_exaggeration,
                                                                     early_exaggeration_iter=early_exaggeration_iter,
                                                                     learning_rate=tsne_learning_rate,
                                                                     max_iter=max_iter,
                                                                     init=init,
                                                                     negative_gradient_method=open_tsne_method,
                                                                     metric=dist_metric)
                FileIO.save_np_array(embedding_open_tsne, dir_path, "open_tsne", col_names=colnames)
                time[0].append("open_tsne")
                time[1].append(time_open_tsne)
            except Exception as e:
                print(e)
                
        if "zifa" in methods:
            try:
                time_zifa, embedding_zifa = LinearMethods.ZIFA(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_zifa, dir_path, "zifa", col_names=colnames)
                time[0].append("ZIFA")
                time[1].append(time_zifa)
            except Exception as e:
                print(e)
                
        if "factor_analysis" in methods:
            try:
                time_factor_analysis, embedding_factor_analysis = LinearMethods.factor_analysis(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_factor_analysis, dir_path, "factor_analysis", col_names=colnames)
                time[0].append("factor_analysis")
                time[1].append(time_factor_analysis)
            except Exception as e:
                print(e)
                
        if "isomap" in methods:
            try:
                time_isomap, embedding_isomap = NonLinearMethods.isomap(data,
                                                                        out_dims=out_dims,
                                                                        transform=transform,
                                                                        dist_metric=dist_metric)
                FileIO.save_np_array(embedding_isomap, dir_path, "isomap", col_names=colnames)
                time[0].append("isomap")
                time[1].append(time_isomap)
            except Exception as e:
                print(e)
                
        if "mds" in methods:
            try:
                time_mds, embedding_mds = LinearMethods.MDS(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_mds, dir_path, "mds", col_names=colnames)
                time[0].append("mds")
                time[1].append(time_mds)
            except Exception as e:
                print(e)
                
        if "lle" in methods:
            try:
                time_lle, embedding_lle = NonLinearMethods.LLE(data,
                                                               out_dims=out_dims,
                                                               transform=transform)
                FileIO.save_np_array(embedding_lle, dir_path, "lle", col_names=colnames)
                time[0].append("lle")
                time[1].append(time_lle)
            except Exception as e:
                print(e)
                
        if "spectral" in methods:
            try:
                time_spectral, embedding_spectral = NonLinearMethods.spectral(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_spectral, dir_path, "spectral", col_names=colnames)
                time[0].append("spectral")
                time[1].append(time_spectral)
            except Exception as e:
                print(e)
                        
        if "kernelpca" in methods:
            try:
                time_kernelPCA, embedding_kernelPCA = NonLinearMethods.kernelPCA(data, out_dims=out_dims, kernel=kernel)
                FileIO.save_np_array(embedding_kernelPCA, dir_path, "kernelPCA", col_names=colnames)
                time[0].append("kernelPCA")
                time[1].append(time_kernelPCA)
            except Exception as e:
                print(e)
                
        if "phate" in methods:
            try:
                time_phate, embedding_phate = NonLinearMethods.phate(data, out_dims=out_dims, decay=phate_decay, knn=phate_knn)
                FileIO.save_np_array(embedding_phate, dir_path, "phate", col_names=colnames)
                time[0].append("phate")
                time[1].append(time_phate)
            except Exception as e:
                print(e)
                
        if "nmf" in methods:
            try:
                time_phate, embedding_phate = LinearMethods.NMF(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_phate, dir_path, "nmf", col_names=colnames)
                time[0].append("nmf")
                time[1].append(time_phate)
            except Exception as e:
                print(e)
                
                
        if "grandprix" in methods:
            try:
                time_phate, embedding_phate = NonLinearMethods.grandprix(data, out_dims=out_dims)
                FileIO.save_np_array(embedding_phate, dir_path, "grandprix", col_names=colnames)
                time[0].append("grandprix")
                time[1].append(time_phate)
            except Exception as e:
                print(e)
                
        FileIO.save_list_to_csv(time, out, "time")
        return time
        

class LinearMethods():
    """Linear DR Methods.

    This class contains static methods of a group of Linear DR Methods.
    """
    
    @staticmethod
    def PCA(data: "np.ndarray",
            out_dims: int=2
            ) -> Tuple[float, "np.ndarray"]:
        
        ''' Scikit-Learn Principal Component Analysis (PCA)
        
        This method uses the SKlearn's standard PCA.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        '''
        
        print("Running PCA.")
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = PCA(n_components=out_dims).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def ICA(data: "np.ndarray",
            out_dims: int,
            max_iter: int=200) -> Tuple[float, "np.ndarray"]:
        
        ''' Scikit-Learn Independent Component Analysis (ICA)
        
        This method uses the SKlearn's FastICA implementation of ICA.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional ICA embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = FastICA(n_components=out_dims, max_iter=max_iter).fit_transform(data) #type:ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def ZIFA(data: "np.ndarray",
             out_dims: int) -> Tuple[float, "np.ndarray"]:
        ''' Zero-Inflated Factor Analysis (ZIFA)
        
        This method implements ZIFA as developed by Pierson & Yau (2015).
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional ICA embedding.
        '''
        
        
        data = LinearMethods._remove_col_zeros(data)
        start_time: float = time.perf_counter()
        
        z: "np.ndarray"
        z, _ = ZIFA.fitModel(data, out_dims)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, z
    
    
    @staticmethod
    def factor_analysis(data: "np.ndarray",
                        out_dims: int):
        ''' Scikit-Learn Factor Analysis (FA)
        
        This method uses the SKlearn's FA implementation.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional ICA embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = FactorAnalysis(n_components=out_dims).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def MDS(data: "np.ndarray",
            out_dims: int,
            metric: bool=True,
            n_jobs: int=-1):
        ''' Scikit-Learn Multi-Dimensional Scaling (MDS)
        
        This method uses the SKlearn's MDS implementation.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            metric (bool): Whether to run metric MDS.
            n_jobs (int): The number of jobs to run concurrantly.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional ICA embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = MDS(n_components=out_dims,
                                      metric=metric,
                                      n_jobs=n_jobs
                                      ).fit_transform(data) #type:ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def NMF(data: "np.ndarray",
            out_dims: int):
        ''' Scikit-Learn Nonnegative Matrix Factorization (NMF)
        
        This method uses the SKlearn's NMF implementation.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional ICA embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = NMF(n_components=out_dims, init = "nndsvd").fit_transform(data) #type:ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def _remove_col_zeros(data: "np.ndarray") -> "np.ndarray":
        ''' Remove a column consisting of entireing zeroes
        
        This private method is used by ZIFA to fix errors.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.

        Returns:
            data (numpy.ndarray): The output array without zero columns.
        '''
        nonzero_col: List[int] = []
        col: int
        for col in range(data.shape[1]):
            if not np.all(data[:,col]==0):
                nonzero_col.append(col)
        return data[:, nonzero_col]
    
    
class NonLinearMethods():
    """NonLinear DR Methods.

    This class contains static methods of a group of NonLinear DR Methods, except for tSNE.
    """
    
    @staticmethod
    def UMAP(data: "np.ndarray",
             out_dims: int=2, 
             n_neighbors: int=15,
             min_dist: float=0.3,
             metric: str="euclidean",
             init: Union["np.ndarray", str]="spectral"
             ) -> Tuple[float, "np.ndarray"]:
        
        ''' UMAP
        
        This method uses the UMAP package's UMAP implementation.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            n_neighbors (int): The number of neighbors to consider.
            min_dist (float): The minimum distance between points in the embedding.
            metric (str): The distance metric used in calculation.
            init (Union[str, "numpy.ndarray"]): Method of initialiazation. 'random', 'spectral', or array.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional UMAP embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = umap.UMAP(n_components=out_dims,
                                            n_neighbors=n_neighbors,
                                            min_dist=min_dist,
                                            metric=metric,
                                            init=init
                                            ).fit_transform(data) #type: ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def saucie(data: "np.ndarray",
               lambda_c: float=0,
               lambda_d: float=0,
               steps: int=1000,
               batch_size: int=256,
               learning_rate: float=0.001
               ) -> Tuple[float, "np.ndarray"]:
        ''' SAUCIE
        
        This method is a wrapper for SAUCIE package's SAUCIE model. Specifically,
        dimension reduction is of interest.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            lambda_c (float): ID regularization.
            lambda_d (float): Within-cluster distance regularization.
            steps (int): The number of training steps to use.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate of traning.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional SUACIE embedding.
        '''
        start_time: float = time.perf_counter()
        
        saucie: "SAUCIE.model.SAUCIE" = SAUCIE.SAUCIE(data.shape[1],
                                                      lambda_c=lambda_c,
                                                      lambda_d=lambda_d,
                                                      learning_rate=learning_rate)
        train: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=True)
        saucie.train(train, steps=steps, batch_size=batch_size)
        
        eval: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=False)
        embedding: "np.ndarray" = saucie.get_embedding(eval) #type: ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def isomap(data: "np.ndarray",
               out_dims: int=2,
               transform: Optional["np.ndarray"]=None,
               n_neighbors: int=5,
               dist_metric: str="euclidean"
               ) -> Tuple[float, "np.ndarray"]:
        
        ''' Isomap
        
        This method is a wrapper for sklearn's implementation of Isomap.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            n_neighbors (int): The number of neighbors to consider.
            dist_metric (str): The distance metric used in calculation.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional Isomap embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        if transform is None:
            embedding: "np.ndarray" = Isomap(n_neighbors=n_neighbors,
                                            n_components=out_dims,
                                            metric=dist_metric,
                                            n_jobs=-1).fit_transform(data)
        else:
            embedding: "np.ndarray" = Isomap(n_neighbors=n_neighbors,
                                             n_components=out_dims,
                                             metric=dist_metric,
                                             n_jobs=-1).fit(data).transform(transform)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def LLE(data: "np.ndarray",
            out_dims: int=2,
            transform: Optional["np.ndarray"]=None,
            n_neighbors: int=5
            ) -> Tuple[float, "np.ndarray"]: 
        
        ''' Locally Linear Embedding (LLE)
        
        This method is a wrapper for sklearn's implementation of LLE.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            n_neighbors (int): The number of neighbors to consider.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional LLE embedding.
        '''
        
        start_time: float = time.perf_counter()

        if transform is None:
            embedding: "np.ndarray" = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                                             n_components=out_dims,
                                                             n_jobs=-1).fit_transform(data)
        else:
            embedding: "np.ndarray" = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                                             n_components=out_dims,
                                                             n_jobs=-1).fit(data).transform(transform)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def kernelPCA(data: "np.ndarray",
                  out_dims: int=2,
                  kernel: str="poly"
                  ) -> Tuple[float, "np.ndarray"]: 
        
        ''' Kernel PCA
        
        This method is a wrapper for sklearn's implementation of kernel PCA.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            kernel (str): The kernel to use: "poly," "linear," "rbf," "sigmoid," or "cosine."

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional kernel PCA embedding.
        '''
        
        start_time: float = time.perf_counter()

        embedding: "np.ndarray" = KernelPCA(n_components=out_dims,
                                            kernel=kernel,
                                            n_jobs=-1).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding


    @staticmethod
    def spectral(data: "np.ndarray",
                 out_dims: int=2):
        
        ''' Spectral Embedding
        
        This method is a wrapper for sklearn's implementation of spectral embedding.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional spectral embedding.
        ''' 
        
        start_time: float = time.perf_counter()

        embedding: "np.ndarray" = SpectralEmbedding(n_components=out_dims,
                                                    n_jobs=-1).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding

    
    @staticmethod
    def phate(data: "np.ndarray",
              out_dims: int=2,
              t: Union[str, int]="auto",
              decay: Optional[int]=40,
              knn: Optional[int]=5):
        ''' PHATE
        
        This method is a wrapper for PHATE.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional PHATE embedding.
        ''' 
        
        start_time: float = time.perf_counter()

        embedding: "np.ndarray" = phate.PHATE(n_components=out_dims,
                                              t=t,
                                              decay=decay,
                                              knn=knn,
                                              n_jobs=-1).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def grandprix(data: "np.ndarray",
                  out_dims: int=2):
        ''' GrandPrix
        
        This method is a wrapper for GrandPrix.
        
        Note:
            The ``GrandPrix`` is not mandatory for this package to run.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional PHATE embedding.
        ''' 
        
        start_time: float = time.perf_counter()

        embedding: "np.ndarray" = GrandPrix.fit_model(data = data, n_latent_dims = out_dims)[0]
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding


class TSNE():
    """Various tSNE implementations.

    This class contains static methods of four different tSNE implementations. In general, we
    recommend the openTSNE implementation for its combination of speed, ease of use, and
    features.
    """
    
    @staticmethod
    def sklearn_tsne(data: "np.ndarray",
                     out_dims: int=2, 
                     perp: int=30,
                     early_exaggeration: float=12.0,
                     learning_rate: float=200.0,
                     max_iter: int=1000,
                     init: Union[str, "np.ndarray"]="random",
                     method: str="barnes_hut",
                     angle: float=0.5,
                     metric: str="euclidean"
                     )-> Tuple[float, "np.ndarray"]:
        
        ''' Scikit-Learn t-SNE
        
        This method uses the Scikit-learn implementation of t-SNE. It supports both
        traditional and BH t-SNE with more control of variables.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            perp (int): Perplexity. The default is set to 30. Tradition is between 30 and 50.
            early_exaggeration (float): The early exaggeration factor of alpha.
            learning_rate (float): The learning rate used during gradient descent.
            max_iter (int): Maximum number of iterations to optimize.
            init (Union[str, "np.ndarray"]): Random ('random') or PCA ('pca') or array initialization. 
            method (str): Original ("exact") or Barnes-Hut ("barnes_hut") implementation.
            angle (float): The speed/accuracy tradeoff for Barnes-Hut t-SNE.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.

        
        '''
        print("Running Scikit-Learn t-SNE: {}".format(method))
        
        start_time: float = time.perf_counter()
        embedding: "np.ndarray" = sklearn.manifold.TSNE(n_components=out_dims, #type: ignore
                                                           perplexity=perp,
                                                           early_exaggeration=early_exaggeration,
                                                           learning_rate=learning_rate,
                                                           n_iter=max_iter,
                                                           init=init,
                                                           method=method,
                                                           angle=angle,
                                                           metric=metric
                                                           ).fit_transform(data)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
    @staticmethod
    def bh_tsne(data: "np.ndarray", 
                out_dims: int=2, 
                perp: int=30,
                theta: float=0.5,
                initial_dims: int=50,
                use_pca: bool=False, 
                max_iter: int=3000
                ) -> Tuple[float, "np.ndarray"]:
        
        ''' Barnes-Hut t-SNE
        
        This method uses the BH t-SNE as proposed and implemented by van der Maaten (2014). As
        opposed to the sklearn implementation, this uses the original implementation.
        
        Note:
            This implementation's installation is not mandatory for this package to run.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            perp: Perplexity. The default is set to 30. Tradition is between 30 and 50.
            theta: The speed/accuracy tradeoff for Barnes-Hut t-SNE.
            initial_dims: Number of dimensions for initial PCA dimension reduction.
            use_pca: Whether to use PCA to first reduce dimension to 50.
            max_iter: Maximum number of iterations to optimize.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        
        '''
        # Learning rate: 200
        print("Running BH t-SNE.")
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = run_bh_tsne(data=data,
                                                 no_dims=out_dims,
                                                 perplexity=perp,
                                                 theta=theta,
                                                 initial_dims=initial_dims,
                                                 use_pca=use_pca,
                                                 max_iter=max_iter)
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
        
        
    @staticmethod
    def fit_sne(data: "np.ndarray",
                out_dims: int=2,
                perp: Union[int, List[int]]=30,
                theta: float=0.5,
                max_iter: int=750,
                stop_early_exag_iter: int=250,
                mom_switch_iter: int=250,
                momentum: float=0.5,
                final_momentum: float=0.8,
                learning_rate: Union[str, float]="auto",
                early_exaggeration: float=12.0,
                no_momentum_during_exag: bool=False,
                n_trees: int=50,
                search_k: Optional[int]=None,
                start_late_exag_iter: Union[int, str]="auto",
                late_exag_coeff: Union[float, int]=-1,
                nterms: int=3,
                intervals_per_integer: int=1,
                min_num_intervals: int=50,
                init: Union[str, "np.ndarray"]="pca",
                load_affinities: Optional[str]=None,
                perplexity_list: Optional[int]=None,
                df: int=1,
                max_step_norm: Optional[float]=5.0,
                ) -> Tuple[float, "np.ndarray"]:
        
        '''FIt-SNE.
        
        This is the FIt-SNE as implemented and introduced by Linderman et al. (2018). It uses interpolation
        and fast fourier transform to accelerate t-SNE.
        
        Note:
            This FIt-SNE implementation is not mandatory for this package to run.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            perp (int): Perplexity. The default is set to 30. Tradition is between 30 and 50.
            theta (float): The speed/accuracy trade-off.
            max_iter (int): Maximum number of iterations to optimize.
            stop_early_exag_iter (int): When to stop early exaggeration.
            mom_switch_iter (int): When to switch momentum.
            momentum (float): Initial value of momentum.
            final_momentum (float): Final value of momentum.
            learning_rate (Union[str, float]): The learning rate used during gradient descent.
            early_exaggeration (float): Early exaggeration factor.
            no_momentum_during_exag (bool): Whether to use momentum during early exaggeration.
            n_trees (int): Number of trees used with Annoy library.
            search_k (Optional[int]): The number of nodes to inspect during search while using Annoy.
            start_late_exag_iter [str]: When to start late exaggeration. 
            late_exag_coeff: Union[float, int] The late exaggeration coefficient to use. Disable with -1.
            nterms (int): The number of interpolation points per inteval.
            intervals_per_integer (int): Used in calculating interpolating intervals.
            min_num_intervals (int): Number of interval for interpolation.
            init (Union[str, "numpy.ndarray"]): Method of initialiazation. 'random', 'pca', or array.
            load_affinities (Optional[str]): Load previous affinities or save them.
            perplexity_list (Optional[int]): List of perplexity for multiperplexity t-SNE.
            df (int): T-distribution degree of freedom.
            max_step_norm (Optional[float]): Maximum step in gradient descent.
            
        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        
        '''
        
        print("Running FIt-SNE.")
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = fast_tsne(X=data,
                                            theta=theta,
                                            perplexity=perp,
                                            map_dims=out_dims,
                                            max_iter=max_iter,
                                            stop_early_exag_iter=stop_early_exag_iter,
                                            mom_switch_iter=mom_switch_iter,
                                            momentum=momentum,
                                            final_momentum=final_momentum,
                                            learning_rate=learning_rate,
                                            early_exag_coeff=early_exaggeration,
                                            no_momentum_during_exag=no_momentum_during_exag,
                                            n_trees=n_trees,
                                            search_k=search_k,
                                            start_late_exag_iter=start_late_exag_iter,
                                            late_exag_coeff=late_exag_coeff,
                                            nterms=nterms,
                                            intervals_per_integer=intervals_per_integer,
                                            min_num_intervals=min_num_intervals,
                                            initialization=init,
                                            load_affinities=load_affinities,
                                            perplexity_list=perplexity_list,
                                            df=df,
                                            max_step_norm=max_step_norm) #type: ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
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
                  negative_gradient_method: str="fft"
                  ) -> Tuple[float, "np.ndarray"]:
        
        '''Open t-SNE.
        
        This is the Python implementation of FIt-SNE through the ``openTSNE`` package. Its implementation
        is based on research from Linderman et al. (2019). This is the default recommended implementation.
        
        Args:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            perp (int): Perplexity. The default is set to 30. Tradition is between 30 and 50.
            learning_rate (Union[str, float]): The learning rate used during gradient descent.
            early_exaggeration_iter (float): Number of early exaggeration iterations.
            early_exaggeration (float): Early exaggeration factor.
            max_iter (int): Maximum number of iterations to optimize.
            dof (int): T-distribution degree of freedom.
            theta (float): The speed/accuracy trade-off.
            init (Union[str, "numpy.ndarray"]): Method of initialiazation. 'random', 'pca', 'spectral', or array.
            
        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        
        '''
        print("Running Open t-SNE.")
        
        n_iter: int = max_iter - early_exaggeration_iter
        start_time: float = time.perf_counter()
        
        affinities_array: Union["affinity.PerplexityBasedNN", "affinity.Multiscale"]
        init_array: "np.ndarray" = np.empty((data.shape[0], out_dims))
        
        if isinstance(perp, list) and len(perp) > 1:
            affinities_array = affinity.Multiscale(data=data,
                                                   perplexities=perp,
                                                   metric=metric,
                                                   n_jobs=-1,
                                                   verbose=True)
        else:
            perp = perp[0] if isinstance(perp, list) else perp
            affinities_array = affinity.PerplexityBasedNN(data=data,
                                                          perplexity=perp,
                                                          metric=metric,
                                                          n_jobs=-1,
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
                                                n_jobs=-1,
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
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding