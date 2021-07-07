import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from annoy import AnnoyIndex

import scipy.spatial
import scipy.stats
import sklearn.metrics

from MEDist.MEDist import approximate
from util import Annoy, DownSample

from typing import Optional, Any, Union, List, Tuple

class Metric():
    
    @classmethod
    def run_metrics_downsample(cls,
                               data: "np.ndarray",
                               embedding: Union["np.ndarray", List["np.ndarray"]],
                               downsample: Optional[int]=None,
                               downsample_indices: Optional[List["np.ndarray"]]=None,
                               n_fold: int=1,
                               methods: Union[str, List[str]]="all",
                               labels: Optional["np.ndarray"]=None,
                               labels_embedding: Optional["np.ndarray"]=None,
                               embedding_names: Optional["np.ndarray"]=None,
                               k: int=5,
                               save_indices_dir: Optional[str]=None,
                               data_annoy_path: Optional[str]=None
                               ) -> List[List[Union[str, float]]]:
        
        '''Run methods with downsampling.
        
        This method first downsamples and runs the methods with the ``run_metrics`` method. This
        is done for efficiency reasons as the current pairwise distance metric is can be memory
        intensive with large datasets.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            downsample (int): The sample size of downsampling.
            downsample_indices (List["np.ndarray], optional): A list of indicies for repeated downsampling.
            n_fold (int): Downsample n times and average the results.
            methods (Union[str, List[str]]): The metrics to run.
            labels ("np.ndarray", optional): True labels or labels from the original space.
            labels_embedding ("np.ndarray", optional): Classification or clustering labels from the embedding space.
            embedding_names ("np.ndarray", optional): Names of the embedding methods to be saved.
            k (int): The number of neighbors for KNN.
            save_indices_dir (str, optional): The directory to save indices generated, if previous indices are not provided.
            data_annoy_path (str, optional): The file path to input data's saved ANNOY model.

        Returns:
            List[List[Union[str, float]]]: A nested list of results with names of metrics, metrics results,
                name of embedding, and downsample index.
        '''
        
        if downsample is None and downsample_indices is None:
            raise ValueError("Either 'downsample' or 'downsample_indices' must be provided.")
        
        if downsample_indices is not None:
            n_fold = len(downsample_indices)
        else:
            assert downsample is not None
            downsample_indices = DownSample.downsample_from_data(data,
                                                                 n=downsample,
                                                                 n_fold=n_fold,
                                                                 save_downsample_index=save_indices_dir)
        
        if not isinstance(embedding, list):
            embedding = [embedding]
            
        index: "np.ndarray"
        data_downsample: "np.ndarray"
        embedding_downsample: Optional[List["np.ndarray"]]=None
        labels_downsample: Optional["np.ndarray"]=None
        labels_embedding_downsample: Optional["np.ndarray"]=None
        results: List[List[Union[str, float]]] = []
        results_downsample_index: List[int] = []
        results_combined: List[List[Any]] = [[],[],[]]
        
        n: int 
        for n in range(n_fold):
            
            index = downsample_indices[n].astype(int)
            data_downsample = data[index, :]

            embedding_downsample = [e[index, :] for e in embedding]
            if labels is not None:
                labels_downsample = labels[index]
            if labels_embedding is not None:
                labels_embedding_downsample = labels_embedding[index]
            
            results = cls.run_metrics(data=data_downsample,
                                      embedding=embedding_downsample,
                                      methods=methods,
                                      labels=labels_downsample,
                                      labels_embedding=labels_embedding_downsample,
                                      embedding_names=embedding_names,
                                      k=k,
                                      data_annoy_path=data_annoy_path)
            
            for col in range(len(results)):
                results_combined[col].extend(results[col])
            results_downsample_index.extend([n]*len(results[0]))

        results_combined.append(results_downsample_index)
        
        return results_combined
        
        
    @classmethod
    def run_metrics(cls,
                    data: "np.ndarray",
                    embedding: Union["np.ndarray", List["np.ndarray"]],
                    methods: Union[str, List[str]]="all",
                    labels: Optional["np.ndarray"]=None,
                    labels_embedding: Optional["np.ndarray"]=None,
                    embedding_names: Optional["np.ndarray"]=None,
                    data_annoy_path: Optional[str]=None,
                    k: int=5
                    ) -> List[List[Union[str, float]]]:
        
        '''Dispatcher to run methods.
        
        This method runs the methods and metrics in this class with the function of
        running multiple or all the metrics. 
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            methods (Union[str, List[str]]): The metrics to run.
            labels ("np.ndarray", optional): True labels or labels from the original space.
            labels_embedding ("np.ndarray", optional): Classification or clustering labels from the embedding space.
            embedding_names ("np.ndarray", optional): Names of the embedding methods to be saved.
            data_annoy_path (str, optional): The path to the saved ANNOY model for input data.
            k (int, optional): The number of neighbors for KNN.

        Returns:
            List[List[Union[str, float]]]: A nested list of results with names of metrics, names of embedding, and metrics results.
        '''
        if isinstance(methods, list):
            methods = [m.lower() for m in methods]
        else:
            methods = [methods.lower()]
            
        if not isinstance(embedding, list):
            embedding = [embedding]
            
        if embedding_names is None:
            embedding_names = np.array(list(map(str, range(len(embedding)))))
        
        i: int
        e: "np.ndarray"    
        data = cls._median_impute(data)
        for i, e in enumerate(embedding):
            embedding[i] = cls._median_impute(e)
            
        if "all" in methods:
            methods = ["knn", "neighborhood_agreement"]
            if labels is not None:
                methods.extend(["npe", "random_forest", "silhouette", "pearsonr", "spearmanr", "residual_variance", "emd"])
            if labels is not None and labels_embedding is not None:
                methods.extend(["ari", "mni"])
                
        data_distance: Optional["np.ndarray"] = None
        embedding_distance: Optional[List["np.ndarray"]] = None
        if any(m in methods for m in ["pearsonr", "spearmanr", "residual_variance", "emd"]):
            assert labels is not None
            data_distance, embedding_distance = cls.pcd_distance(data, embedding, labels, )
        
        annoy_data_neighbors: Optional["np.ndarray"] = None
        annoy_embedding_neighbors: Optional[List["np.ndarray"]] = None
        if any(m in methods for m in ["knn", "npe", "neighborhood_agreement"]):
            annoy_data_neighbors, annoy_embedding_neighbors = cls.build_annoy(data, embedding, data_annoy_path, k)

        results: List[List[Any]] = [[],[],[]]
        
        for i in range(embedding_names.shape[0]):
            
            e = embedding[i]
            e = cls._median_impute(e)
             
            if "pearsonr" in methods:
                print("running pearsonr")
                assert data_distance is not None and embedding_distance is not None
                cor: float=cls.correlation(x=data_distance, y=embedding_distance[i], metric="Pearson")
                results[0].append("correlation_pearson")
                results[1].append(cor)
                results[2].append(embedding_names[i])
                
            if "spearmanr" in methods:
                print("running spearmanr")
                assert data_distance is not None and embedding_distance is not None
                cor: float=cls.correlation(x=data_distance, y=embedding_distance[i], metric="Spearman")
                results[0].append("correlation_spearman")
                results[1].append(cor)
                results[2].append(embedding_names[i])
                
            if "residual_variance" in methods:
                print("running residual variance")
                # TODO: Implement r when pearsonr is not None.
                assert data_distance is not None and embedding_distance is not None
                results[0].append("residual_variance")
                results[1].append(cls.residual_variance(x=data_distance, y=embedding_distance[i]))
                results[2].append(embedding_names[i])
                
            if "knn" in methods:
                print("running knn")
                assert annoy_data_neighbors is not None and annoy_embedding_neighbors is not None
                results[0].append("knn")
                results[1].append(cls.KNN(data_neighbors=annoy_data_neighbors, embedding_neighbors=annoy_embedding_neighbors[i]))
                results[2].append(embedding_names[i])
            
            if "npe" in methods:
                print("running npe")
                assert annoy_data_neighbors is not None and annoy_embedding_neighbors is not None
                assert labels is not None
                results[0].append("npe")
                results[1].append(cls.NPE(labels = labels, data_neighbors=annoy_data_neighbors, embedding_neighbors=annoy_embedding_neighbors[i]))
                results[2].append(embedding_names[i])
                
            if "neighborhood_agreement" in methods:
                print("running neighborhood agreement")
                assert annoy_data_neighbors is not None and annoy_embedding_neighbors is not None
                results[0].append("neighborhood_agreement")
                results[1].append(cls.neighborhood_agreement(data_neighbors=annoy_data_neighbors, embedding_neighbors=annoy_embedding_neighbors[i]))
                results[2].append(embedding_names[i])
                
            if "emd" in methods:
                print("running emd")
                assert data_distance is not None and embedding_distance is not None
                results[0].append("emd")
                results[1].append(cls.EMD(x=data_distance, y=embedding_distance[i]))
                results[2].append(embedding_names[i])
                
            if "random_forest" in methods:
                print("running random_forest")
                assert labels is not None and e is not None
                results[0].append("random_forest")
                results[1].append(cls.random_forest(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "silhouette" in methods:
                print("running silhouette")
                assert labels is not None and e is not None
                results[0].append("silhouette")
                results[1].append(cls.silhouette(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "nmi" in methods:
                print("running mni")
                assert labels is not None and labels_embedding is not None
                results[0].append("nmi")
                results[1].append(cls.NMI(labels=labels, labels_embedding=labels_embedding))
                results[2].append(embedding_names[i])
                
            if "ari" in methods:
                print("running ari")
                assert labels is not None and labels_embedding is not None
                results[0].append("ari")
                results[1].append(cls.ARI(labels=labels, labels_embedding=labels_embedding))
                results[2].append(embedding_names[i])
            
        return results
    
    
    @staticmethod
    def _median_impute(data: "np.ndarray"):
        nan: "np.ndarray" = np.unique(np.where(np.isnan(data))[0])
        
        if nan.size == 0:
            return data
        else:
            median: "np.ndarray" = np.nanmedian(data, axis=0)
            print(median)
            i: int
            for i in nan:
                data[i] = median
            print(np.where(np.isnan(data)))
            return data
    
    
    @staticmethod
    def pcd_distance(data: "np.ndarray",
                     embedding: List["np.ndarray"],
                     labels: "np.ndarray"
                     ) -> Tuple["np.ndarray", List["np.ndarray"]]:
        
        '''Calculate Point Cluster Distanced (PCD).
        
        Utility wrapper method to compute the Point Cluster Distance. The point cluster distance computes the
        distance between each point and each cluster's centroid.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            labels ("np.ndarray"): True labels or labels from the original space.

        Returns:
            Tuple["np.ndarray", List["np.ndarray"]]: A tuple with PCD of original space data and lower dimension embedding.
        '''
        
        data_distance: "np.ndarray"
        embedding_distance: List["np.ndarray"]
        
        pcd_data: "approximate.PointClusterDistance" = approximate.PointClusterDistance(X=data, labels=labels)
        data_distance = pcd_data.fit(flatten=True)[0]
        embedding_distance = []
        for e in embedding:
            pcd_embedding: "approximate.PointClusterDistance" = approximate.PointClusterDistance(X=e, labels=labels)
            embedding_distance.append(pcd_embedding.fit(flatten=True)[0])
            
        return data_distance, embedding_distance
        
    
    @staticmethod
    def build_annoy(data: "np.ndarray",
                    embedding: List["np.ndarray"],
                    data_annoy_path: Optional[str]=None,
                    k: int=5) -> Tuple["np.ndarray", List["np.ndarray"]]:
        
        '''Build ANNOY and returns nearest neighbors.
        
        This is a utility function for building ANNOY models and returning the nearest-neighbor matrices for original
        space data and low-dimensional embedding. 
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            data_annoy_path (str): The path to pre-built ANNOY model for original data.

        Returns:
            Tuple["np.ndarray", List["np.ndarray"]]: A tuple with nearest-neighbor matrices of original space data and lower dimension embedding.
        '''
        
        if data_annoy_path is not None:
            annoy_model_data = Annoy.load_annoy(path=data_annoy_path, ncol=data.shape[1])
        else:
            annoy_model_data = Annoy.build_annoy(data)
            
        annoy_data_neighbors = np.empty((data.shape[0], k), dtype=int)
        for i in range(data.shape[0]):
            data_neighbors: List[int] = annoy_model_data.get_nns_by_item(i, k+1)
            # Try remove the i-th obs itself
            try:
                data_neighbors.remove(i)
            except ValueError:
                data_neighbors = data_neighbors[0:(len(data_neighbors)-1)]
                
            annoy_data_neighbors[i] = data_neighbors
        
        annoy_embedding_neighbors = []
        for e in embedding:
            annoy_model_embedding: "AnnoyIndex" = Annoy.build_annoy(e)
            e_neighbors: "np.ndarray" = np.empty((e.shape[0], k), dtype=int)
            for i in range(e.shape[0]):
                embedding_neighbors: List[int] = annoy_model_embedding.get_nns_by_item(i, k+1)
                
                try:
                    embedding_neighbors.remove(i)
                except ValueError:
                    embedding_neighbors = embedding_neighbors[0:(len(embedding_neighbors)-1)]
                    
                e_neighbors[i] = embedding_neighbors
            annoy_embedding_neighbors.append(e_neighbors)
        
        return annoy_data_neighbors, annoy_embedding_neighbors
        
    
    @staticmethod
    def correlation(x: "np.ndarray",
                    y: "np.ndarray",
                    metric: str="Pearson") -> float:
        
        '''Correlation
        
        This method computes the pearson or spearman correlation between the inputs.
        
        Parameters:
            x (np.ndarray): The first 1D array. 
            y (np.ndarray): The second 1D array.
            metric (str): The metric to use. 'Pearson' or 'Spearman'.

        Returns:
            float: Correlation.
        '''
        
        if metric.lower() == "pearson":
            cor: float=scipy.stats.pearsonr(x, y)[0]
        elif metric.lower() == "spearman":
            cor: float=scipy.stats.spearmanr(x, y)[0]
        else:
            raise ValueError("Unsupported metric: must be 'Pearson' or 'Spearman.'")
        
        return cor
    
    
    @staticmethod
    def residual_variance(x: Optional["np.ndarray"]=None,
                          y: Optional["np.ndarray"]=None,
                          r: Optional[float]=None) -> float:
        
        '''Residual Variance
        
        The residual variance is computed with the following formuation with r as the
        pearson correlation: 1-r**2. If r is provided, x and y are optional for efficiency.
        
        Parameters:
            x (np.ndarray, optional): The first 1D array. 
            y (np.ndarray, optional): The second 1D array.
            r (float, optional): Pearson correlation between x and y.

        Returns:
            float: Redisual variance.
        '''
        
        if r is None:
            if x is None or y is None:
                raise ValueError("Either 'r' or both 'x' and 'y' is needed.")
            else:
                cor = Metric.correlation(x, y, metric="Pearson")
                return 1 - cor**2
        elif r > 1 or r < -1:
            raise ValueError("'r' must be between -1 and 1.")
        else:
            return 1 - r**2
        
    @staticmethod
    def EMD(x: "np.ndarray",
            y: "np.ndarray") -> float:
        
        '''Earth Mover's Distance (EMD)
        
        This metric computes the EMD between the pairwise distance of between points in the
        high and low dimensional space. This implementation uses the ``scipy.stats.wasserstein_distance``.
        The usage of EMD is proposed in Heiser & Lou (2020).
        
        Parameters:
            x (np.ndarray): The first distribution x as a 1D array.
            y (np.ndarray): The second distribution y as a 1D array.

        Returns:
            float: Earth mover's distance.
        
        '''
        return scipy.stats.wasserstein_distance(x, y)

    
    @staticmethod
    def KNN(data_neighbors: "np.ndarray",
            embedding_neighbors: "np.ndarray") -> float:
        
        '''K-Nearest Neighbors Preservation (KNN)
        
        The KNN metric computes the percentage of k-neighbors of each point is preserved in the
        embedding space, and it is average across the entire dataset.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            k (int, optional): The number of neighbors for KNN. This is required when either knn_model_data
                or knn_model_embedding is not supplied.
            knn_model_data (sklearn.neighbors.NearestNeighbors, optional):
                A fitted instance of ``sklearn.neighbors.NearestNeighbors`` with ``data``.
            knn_model_embedding (sklearn.neighbors.NearestNeighbors, optional): 
                A fitted instance of ``sklearn.neighbors.NearestNeighbors`` with ``embedding``.

        Returns:
            float: K-nearest neighbors preservation.
        '''
        
        i: int
        intersect: int = 0
        for i in range(data_neighbors.shape[0]):
            intersect += np.intersect1d(data_neighbors[i], embedding_neighbors[i]).shape[0]
        
        return intersect/data_neighbors.size

    
    @staticmethod
    def NPE(data_neighbors: "np.ndarray",
            embedding_neighbors: "np.ndarray",
            labels: "np.ndarray") -> float:
        
        '''Neighborhood Proportion Error (NPE)
        
        The NPE metric is proposed by Konstorum et al. (2019). It measures the total variation distance between
        the proportion of nearest points belonging to the same class of each point in the HD and LD space. The
        lower the NPE, the more similar the embedding and the original data are.
        
        Parameters:
            data_neighbors (np.ndarray): A nearest-neighbor matrix of the original data.
            embedding_neighbors (np.ndarray): A nearest-neighbor matrix of the embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Neighborhood proportion error.
        '''
        
        classes: "np.ndarray"
        classes_index: "np.ndarray"
        classes, classes_index = np.unique(labels, return_inverse=True)
        
        same_class_data: "np.ndarray" = np.zeros(data_neighbors.shape[0])
        same_class_embedding: "np.ndarray" = np.zeros(data_neighbors.shape[0]) 
        
        k: int = data_neighbors.shape[1]
        i: int 
        for  i in range(data_neighbors.shape[0]):
            i_index = classes_index[i]
            i_neighbors_data = data_neighbors[i]
            i_neighbors_embedding = embedding_neighbors[i]
            
            same_class_data[i] = np.count_nonzero(classes_index[i_neighbors_data]==i_index)/k
            same_class_embedding[i] = np.count_nonzero(classes_index[i_neighbors_embedding]==i_index)/k
        
        distance: float=0.0
        c: Any
        for c in classes:
            P: "np.ndarray" = same_class_data[classes_index==c]
            Q: "np.ndarray" = same_class_embedding[classes_index==c]
            
            distance += np.sum(np.absolute(P-Q))/2
            
        return distance/classes.size
    
    
    @staticmethod
    def neighborhood_agreement(data_neighbors: "np.ndarray",
                               embedding_neighbors: "np.ndarray") -> float:
        '''Neighborhood Agreement
        
        The Neighborhood Agreement metric is proposed by Lee et al. (2015). It measures
        the intersection of k-nearest neighbors (KNN) of each point in HD and LD space. The
        result is subsequently rescaled to measure the improvement over a random embedding.
        This measure is conceptually similar to ``Metric.KNN`` such that they both measure
        the agreement of KNN, but ``Metric.KNN`` simply takes the average of the KNN graph
        agreement without any scaling.
        
        Parameters:
            data_neighbors (np.ndarray): A nearest-neighbor matrix of the original data.
            embedding_neighbors (np.ndarray): A nearest-neighbor matrix of the embedding.

        Returns:
            float: Neighborhood agreement.
        
        '''
        
        k: int = data_neighbors.shape[1]
        i: int
        agreement: float = 0.0
        for i in range(data_neighbors.shape[0]):
            agreement += np.intersect1d(data_neighbors[i], embedding_neighbors[i]).shape[0]
        
        agreement = (agreement/(k*data_neighbors.shape[0])*(data_neighbors.shape[0]-1)-k)/(data_neighbors.shape[0]-1-k)
        
        return agreement
    
    
    @staticmethod
    def neighborhood_trustworthiness(data_neighbors: "np.ndarray",
                                     embedding_neighbors: "np.ndarray",
                                     dist_data: "np.ndarray") -> float:
        '''Neighborhood Trustworthiness 
        
        The Neighborhood Truestworthiness is proposed by Venna and Kaski (2001). It measures
        trustworthiness by measuring the ranked distane of new points entering the defined
        neighborhood size in the embedding. The higher the new points are ranked based on the
        original HD space distance matrix, the less trustworthy the new embedding is. The measure
        is scaled between 0 and 1 with a higher score reflecting a more trustworthy embedding.
        
        Parameters:
            data_neighbors (np.ndarray): A nearest-neighbor matrix of the original data.
            embedding_neighbors (np.ndarray): A nearest-neighbor matrix of the embedding.
            dist_data (np.ndarray): A pairwise distance matrix for the original data.

        Returns:
            float: Neighborhood trustworthiness.
        
        '''
        
        dist_data = scipy.spatial.distance.squareform(dist_data)
        dist_data = scipy.stats.rankdata(dist_data, axis=1)
        
        k: int = data_neighbors.shape[1]
        score: float = 0
        i: int
        for i in range(data_neighbors.shape[0]):
            neighbor_diff: "np.ndarray" = np.setdiff1d(embedding_neighbors[i], data_neighbors[i], assume_unique=True)
            score += np.sum(dist_data[i, neighbor_diff] - k)
            
        score = 1 - 2*score/(data_neighbors.shape[0]*k*(2*data_neighbors.shape[0]-3*k-1))
        
        return score
    
    
    @staticmethod
    def random_forest(embedding: "np.ndarray",
                      labels: "np.ndarray") -> float:
        
        '''Random Forest Classification Accuracy
        
        This function trains a random forest classifer using the embedding data and the labels
        generated or manually classified from the original space. It then tests the accuracy
        of the classifier using the 33% of the embedding data. This metric was first proposed in
        Becht et al. (2019).
        
        Parameters:
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Random forest accuracy.
        
        '''
        
        embedding_train, embedding_test, labels_train, labels_test = train_test_split(embedding, labels, test_size=0.33)

        rf: "RandomForestClassifier" = RandomForestClassifier().fit(embedding_train, labels_train)
        predictions: "np.ndarray" = rf.predict(embedding_test)
        
        return sklearn.metrics.accuracy_score(labels_test, predictions)
    
    
    @staticmethod
    def silhouette(embedding: "np.ndarray",
                   labels: "np.ndarray") -> float:
        
        '''Silhouette Score
        
        This metric computes the silhouette score of clusters in the embedding space. Ideally,
        clusters should be coherent, and using labels obtained from the original space can
        evaluate the effectiveness of the embedding technique. This metric is used in 
        Xiang et al. (2021).
        
        Parameters:
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Silhouette score.
        
        '''
        
        return sklearn.metrics.silhouette_score(embedding, labels)
    
    
    @staticmethod
    def NMI(labels: "np.ndarray",
            labels_embedding: "np.ndarray") -> float:
        
        '''Normalized Mutual Information (NMI)
        
        The NMI metric computes the mutual information between labels of the original space
        and the embeeding space and then normalizes it with the larger entroy of the two vectors.
        This metric is a measure of clustering performance before and after dimension reduction,
        and it is used in Xiang et al. (2021).
        
        Parameters:
            labels (np.ndarray): The class labels of of the original space.
            labels_embedding (np.ndarray): The class labels generated from the embedding space.

        Returns:
            float: Silhouette score.
        '''
        
        mi: float = sklearn.metrics.mutual_info_score(labels, labels_embedding)
        
        labels_count: "np.ndarray"
        embedding_count: "np.ndarray"
        
        _ , labels_count = np.unique(labels, return_counts=True)
        _ , embedding_count = np.unique(labels_embedding, return_counts=True)
        
        labels_count = labels_count/labels.size
        embedding_count = embedding_count/labels_embedding.size
        normalization: float = np.maximum(scipy.stats.entropy(labels_count),
                                          scipy.stats.entropy(embedding_count))
        
        return mi/normalization
    
    
    @staticmethod
    def ARI(labels: "np.ndarray",
            labels_embedding: "np.ndarray") -> float:
        
        '''Adjusted Rand Index (ARI)
        
        The ARI uses the labels from the original space and the embedding space to measure
        the similarity between them using pairs. It is used in Xiang et al. (2021).
        
        Parameters:
            labels (np.ndarray): The class labels of of the original space.
            labels_embedding (np.ndarray): The class labels generated from the embedding space.

        Returns:
            float: ARI.
            
        References:
            This implementation adapts from sklearn's implementation of ARI with a bug fix of overflow
            issue. 
            
            @article{scikit-learn,
            title={Scikit-learn: Machine Learning in {P}ython},
            author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
                    and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
                    and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
                    Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
            journal={Journal of Machine Learning Research},
            volume={12},
            pages={2825--2830},
            year={2011}
            }
            
        Licenses:
        
            BSD 3-Clause License

            Copyright (c) 2007-2021 The scikit-learn developers.
            All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are met:

            * Redistributions of source code must retain the above copyright notice, this
            list of conditions and the following disclaimer.

            * Redistributions in binary form must reproduce the above copyright notice,
            this list of conditions and the following disclaimer in the documentation
            and/or other materials provided with the distribution.

            * Neither the name of the copyright holder nor the names of its
            contributors may be used to endorse or promote products derived from
            this software without specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
            FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
            CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
            OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        '''
        
        confusion: "np.ndarray" = sklearn.metrics.pair_confusion_matrix(labels, labels_embedding)
        
        tn: int = int(confusion[0][0])
        fp: int = int(confusion[0][1])
        fn: int = int(confusion[1][0])
        tp: int = int(confusion[1][1])
        
        return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))