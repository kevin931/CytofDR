import numpy as np
import sklearn
from sklearn.decomposition import FastICA, PCA

import umap
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

from main import FileIO

import time
import os
from typing import Tuple, Union, Optional, List, Dict

METHODS: Dict[str, bool]  = {"fit_sne": True, "bh_tsne": True, "saucie": True}

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
    

class DR():

    @classmethod
    def run_methods(cls,
                    data: "np.ndarray",
                    out: str,
                    methods: Union[str, List[str]]="all",
                    out_dims: int=2, 
                    perp: Union[int, List[int]]=30,
                    early_exaggeration: float=12.0,
                    early_exaggeration_iter: int=250,
                    tsne_learning_rate: Optional[Union[float, str]]=None,
                    max_iter: int=1000,
                    init: str="random",
                    open_tsne_method="fft"
                    ) -> List[List[Union[str, float]]]:
        
        dir_path: str = out+"/embedding"
        os.mkdir(dir_path)
        
        if not isinstance(methods, list):
            methods = [methods]
        methods = [each_method.lower() for each_method in methods]
        
        if "all" in methods:
            methods = ["pca", "umap", "sklearn_tsne_original", "sklearn_tsne_bh", "open_tsne"]
            if METHODS["fit_sne"]:
                methods.append("fit_sne")
            if METHODS["bh_tsne"]:
                methods.append("bh_tsne")
            if METHODS["saucie"]:
                methods.append("saucie")
                
        if not isinstance(perp, list):
            perp = [perp]

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
                FileIO.save_np_array(embedding_PCA, dir_path, "PCA")
                time[0].append("PCA")
                time[1].append(time_PCA)
            except Exception as e:
                print(e)
                
        if "ica" in methods:
            try:
                time_ICA, embedding_ICA = LinearMethods.ICA(data, out_dim=out_dims)
                FileIO.save_np_array(embedding_ICA, dir_path, "ICA")
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
                    
                time_UMAP, embedding_UMAP = NonLinearMethods.UMAP(data, out_dims=out_dims, init=init_umap)
                FileIO.save_np_array(embedding_UMAP, dir_path, "UMAP")
                time[0].append("UMAP")
                time[1].append(time_UMAP)
            except Exception as e:
                print(e)
                
        if "saucie" in methods:
            try:
                time_saucie, embedding_saucie = NonLinearMethods.saucie(data)
                FileIO.save_np_array(embedding_saucie, dir_path, "SAUCIE")
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
                                                                                init=init)
                FileIO.save_np_array(embedding_tsne_original, dir_path, "tsne_original")
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
                                                                                    init=init)
                FileIO.save_np_array(embedding_sklearn_tsne_bh, dir_path, "sklearn_tsne_bh")
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
                FileIO.save_np_array(embedding_bh_tsne, dir_path, "bh_tsne")
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
                                                                max_iter=max_iter,
                                                                perplexity_list=perp_list, #type:ignore
                                                                init=init) 
                FileIO.save_np_array(embedding_fit_sne, dir_path, "fit_sne")
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
                                                                        max_iter=max_iter,
                                                                        init=init,
                                                                        negative_gradient_method=open_tsne_method)
                FileIO.save_np_array(embedding_open_tsne, dir_path, "open_tsne")
                time[0].append("open_tsne")
                time[1].append(time_open_tsne)
            except Exception as e:
                print(e)
                
        FileIO.save_list_to_csv(time, out, "time")
            
        return time
        

class LinearMethods():
    
    @staticmethod
    def PCA(data: "np.ndarray",
            out_dims: int=2
            ) -> Tuple[float, "np.ndarray"]:
        
        ''' Scikit-Learn Principal Component Analysis
        
        This method uses the SKlearn's standard PCA.
        
        Parameters:
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
            out_dim: int,
            max_iter: int=200):
        
        ''' Scikit-Learn Independent Component Analysis
        
        This method uses the SKlearn's FastICA implementation of ICA.
        
        Parameters:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
        '''
        
        start_time: float = time.perf_counter()
        
        embedding: "np.ndarray" = FastICA(n_components=out_dim, max_iter=max_iter).fit_transform(data) #type:ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding
    
    
class NonLinearMethods():
    
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
        
        Parameters:
            data (numpy.ndarray): The input high-dimensional array.
            out_dims (int): The number of dimensions of the output.
            n_neighbors (int): The number of neighbors to consider.
            min_dist (float): The minimum distance between points in the embedding.
            metric (str): The distance metric used in calculation.
            init (Union[str, "numpy.ndarray"]): Method of initialiazation. 'random', 'spectral', or array.

        Returns:
            run_time (float): Time used to produce the embedding.
            embedding (numpy.ndarray): The low-dimensional t-SNE embedding.
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
    def saucie(data: "np.ndarray") -> Tuple[float, "np.ndarray"]:
        
        start_time: float = time.perf_counter()
        
        saucie: "SAUCIE.model.SAUCIE" = SAUCIE.SAUCIE(data.shape[1])
        train: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=True)
        saucie.train(train, steps=1000)
        
        eval: "SAUCIE.loader.Loader" = SAUCIE.Loader(data, shuffle=False)
        embedding: "np.ndarray" = saucie.get_embedding(eval) #type: ignore
        
        end_time: float = time.perf_counter()
        run_time: float = end_time - start_time
        
        return run_time, embedding


class TSNE():
    
    @staticmethod
    def sklearn_tsne(data: "np.ndarray",
                     out_dims: int=2, 
                     perp: int=30,
                     early_exaggeration: float=12.0,
                     learning_rate: float=200.0,
                     max_iter: int=1000,
                     init: Union[str, "np.ndarray"]="random",
                     method: str="barnes_hut",
                     angle: float=0.5
                     )-> Tuple[float, "np.ndarray"]:
        
        ''' Scikit-Learn t-SNE
        
        This method uses the Scikit-learn implementation of t-SNE. It supports both
        traditional and BH t-SNE with more control of variables.
        
        Parameters:
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
                                                           angle=angle
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
        
        This method uses the BH t-SNE as proposed and implemented by van der Maaten (2014).
        
        Parameters:
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
        
        Parameters:
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
                  distance_metric: str="euclidean",
                  dof: int=1,
                  theta: float=0.5, 
                  init: Union["np.ndarray", str]="pca",
                  negative_gradient_method: str="fft"
                  ) -> Tuple[float, "np.ndarray"]:
        
        '''Open t-SNE.
        
        This is the Python implementation of FIt-SNE through the openTSNE package. Its implementation
        is based on research from Linderman et al. (), 
        
        Parameters:
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
                                                   metric=distance_metric,
                                                   n_jobs=-1,
                                                   verbose=True)
        else:
            perp = perp[0] if isinstance(perp, list) else perp
            affinities_array = affinity.PerplexityBasedNN(data=data,
                                                          perplexity=perp,
                                                          metric=distance_metric,
                                                          n_jobs=-1,
                                                          verbose=True)

        if isinstance(init, str):
            if init == "pca":
                init_array = initialization.pca(data, n_components=out_dims)
            elif init == "spectral":
                init_array = initialization.spectral(data, n_components=out_dims)
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