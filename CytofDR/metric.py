import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from annoy import AnnoyIndex

import scipy.spatial
import scipy.stats
import sklearn.metrics

from CytofDR.util import Annoy
import itertools
from typing import Optional, Any, Union, List, Tuple, Callable

class Metric():
    
    @classmethod
    def run_metrics_downsample(cls,
                               data: "np.ndarray",
                               embedding: Union["np.ndarray", List["np.ndarray"]],
                               downsample_indices: List["np.ndarray"],
                               methods: Union[str, List[str]]="all",
                               pwd_metric: str="PCD",
                               labels: Optional["np.ndarray"]=None,
                               labels_embedding: Optional["np.ndarray"]=None,
                               comparison_file: Optional[Union["np.ndarray", List["np.ndarray"]]]=None,
                               comparison_labels: Optional[Union["np.ndarray", List["np.ndarray"]]]=None,
                               comparison_classes: Optional[Union[str, List[str]]]=None,
                               embedding_names: Optional["np.ndarray"]=None,
                               k: int=5,
                               data_annoy_path: Optional[str]=None
                               ) -> List[List[Union[str, float]]]:
        
        '''Run evaluation metrics with downsampling.
        
        This method first downsamples and runs the methods with the ``run_metrics`` method. This
        is done for efficiency reasons as the current pairwise distance metric is can be memory
        intensive with large datasets.
        
        Args:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            downsample (int): The sample size of downsampling.
            downsample_indices (List["np.ndarray], optional): A list of indicies_embedding for repeated downsampling.
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
        for n in range(len(downsample_indices)):
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
                                      pwd_metric=pwd_metric,
                                      labels=labels_downsample,
                                      labels_embedding=labels_embedding_downsample,
                                      comparison_file=comparison_file,
                                      comparison_labels=comparison_labels,
                                      comparison_classes=comparison_classes,
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
                    pwd_metric: str="PCD",
                    labels: Optional["np.ndarray"]=None,
                    labels_embedding: Optional["np.ndarray"]=None,
                    comparison_file: Optional[Union["np.ndarray", List["np.ndarray"]]]=None,
                    comparison_labels: Optional[Union["np.ndarray", List["np.ndarray"]]]=None,
                    comparison_classes: Optional[Union[str, List[str]]]=None,
                    embedding_names: Optional["np.ndarray"]=None,
                    data_annoy_path: Optional[str]=None,
                    k: int=5
                    ) -> List[List[Union[str, float]]]:
        
        '''Run evaluation metrics.
        
        This is an all-in-one method as a dispatcher that supports running multiple metrics
        with one function call.
        
        Args:
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
            
        if pwd_metric.lower() != "pcd" or labels is None:
            pwd_metric = "pairwise"
        
        i: int
        e: "np.ndarray"    
        data = cls._median_impute(data)
        for i, e in enumerate(embedding):
            embedding[i] = cls._median_impute(e)
            
        if "all" in methods:
            methods = ["knn", "neighborhood_agreement", "npe", "random_forest", "silhouette", "pearsonr", "spearmanr", "residual_variance", "emd"]
            if labels is not None:
                methods.extend(["npe", "random_forest", "silhouette"])
            if labels is not None and labels_embedding is not None:
                methods.extend(["ari", "nmi"])
            if comparison_labels is not None and comparison_file is not None:
                methods.extend(["embedding_concordance"])
                
        data_distance: Optional["np.ndarray"] = None
        embedding_distance: Optional[List["np.ndarray"]] = None
        if any(m in methods for m in ["pearsonr", "spearmanr", "residual_variance", "emd"]):
            if pwd_metric.lower() == "pcd":
                assert labels is not None
                data_distance, embedding_distance = cls.pcd_distance(data, embedding, labels,  dist_metric="euclidean")
                
            else:
                data_distance = scipy.spatial.distance.pdist(data)
                embedding_distance = []
                for e in embedding:
                    embedding_distance.append(scipy.spatial.distance.pdist(e, metric="euclidean"))
        
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
                print("running nmi")
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
                
            if "calinski_harabasz" in methods:
                print("running calinski_harabasz")
                assert labels is not None and e is not None
                results[0].append("calinski_harabasz")
                results[1].append(cls.calinski_harabasz(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "davies_bouldin" in methods:
                print("running davies_bouldin")
                assert labels is not None and e is not None
                results[0].append("davies_bouldin")
                results[1].append(cls.davies_bouldin(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "concordance_emd" in methods:
                print("running concordance_emd")
                assert labels_embedding is not None
                assert comparison_file is not None and comparison_labels is not None
                results[0].append("concordance_emd")
                results[1].append(cls.embedding_concordance(e, labels_embedding, comparison_file, comparison_labels, comparison_classes, "emd"))
                results[2].append(embedding_names[i])
                
            if "concordance_cluster_distance" in methods:
                print("running concordance_cluster_distance")
                assert labels_embedding is not None
                assert comparison_file is not None and comparison_labels is not None
                results[0].append("concordance_cluster_distance")
                results[1].append(cls.embedding_concordance(e, labels_embedding, comparison_file, comparison_labels, comparison_classes, "cluster_distance"))
                results[2].append(embedding_names[i])
            
        return results
    
    
    @staticmethod
    def _median_impute(data: "np.ndarray") -> "np.ndarray":
        '''Median imputation for NaN.
                
        Utility method to fix any NaN in datsets with median imputation. When there is no
        NaN, the original array is returned.
        
        Args:
            data (np.ndarray): The input high-dimensional array.

        Returns:
           np.ndarray: An array without NaN.
        '''
        nan: "np.ndarray" = np.unique(np.where(np.isnan(data))[0])
        
        if nan.size == 0:
            return data
        else:
            median: "np.ndarray" = np.nanmedian(data, axis=0)
            i: int
            for i in nan:
                data[i] = median
            return data
    
    
    @staticmethod
    def pcd_distance(data: "np.ndarray",
                     embedding: List["np.ndarray"],
                     labels: "np.ndarray",
                     dist_metric: "str" = "euclidean"
                     ) -> Tuple["np.ndarray", List["np.ndarray"]]:
        
        '''Calculate Point Cluster Distanced (PCD).
        
        Utility wrapper method to compute the Point Cluster Distance. The point cluster distance computes the
        distance between each point and each cluster's centroid.
        
        Args:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            labels ("np.ndarray"): True labels or labels from the original space.

        Returns:
            Tuple["np.ndarray", List["np.ndarray"]]: A tuple with PCD of original space data and lower dimension embedding.
        '''
        
        data_distance: "np.ndarray"
        embedding_distance: List["np.ndarray"]
        
        pcd_data: "PointClusterDistance" = PointClusterDistance(X=data, labels=labels, dist_metric=dist_metric)
        data_distance = pcd_data.fit(flatten=True)[0]
        embedding_distance = []
        for e in embedding:
            pcd_embedding: "PointClusterDistance" = PointClusterDistance(X=e, labels=labels, dist_metric="euclidean")
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
        
        Args:
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
        '''Calculate Correlation Coefficient
        
        This method computes the pearson or spearman correlation between the inputs.
        
        Args:
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
        
        Args:
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
        
        Args:
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
        
        Note:
            This method is not used to calculate KNN itself.
        
        Args:
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
        
        Args:
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
            P: "np.ndarray" = same_class_data[labels==c]
            Q: "np.ndarray" = same_class_embedding[labels==c]
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
        
        Args:
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
        
        Args:
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
        
        This method trains a random forest classifer using the embedding data and the labels
        generated or manually classified from the original space. It then tests the accuracy
        of the classifier using the 33% of the embedding data. This metric was first proposed in
        Becht et al. (2019).
        
        Args:
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
        
        Args:
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
        
        Args:
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
        
        Args:
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
    
    
    @staticmethod
    def calinski_harabasz(embedding: "np.ndarray",
                          labels: "np.ndarray") -> float:
        '''Calinski-Harabasz Index
        
        This metric computes the Calinski-Harabasz index of clusters in the embedding space. Ideally,
        clusters should be coherent, and using labels obtained from the original space can
        evaluate the effectiveness of the embedding technique.
        
        Args:
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Calinski-Harabasz Index.
        '''
        
        return sklearn.metrics.calinski_harabasz_score(embedding, labels)
    
    
    @staticmethod
    def davies_bouldin(embedding: "np.ndarray",
                       labels: "np.ndarray") -> float:
        '''Davies-Bouldin Index
        This metric computes the Davies-Bouldin index of clusters in the embedding space. Ideally,
        clusters should be coherent, and using labels obtained from the original space can
        evaluate the effectiveness of the embedding technique.
        
        Args:
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Davies-Bouldin Index.
        
        '''
        
        return sklearn.metrics.davies_bouldin_score(embedding, labels)
    
    
    @staticmethod
    def embedding_concordance(embedding: "np.ndarray",
                              labels_embedding: "np.ndarray",
                              comparison_file: Union["np.ndarray", List["np.ndarray"]],
                              comparison_labels: Union["np.ndarray", List["np.ndarray"]],
                              comparison_classes: Optional[Union[str, List[str]]]=None,
                              method: str = "emd"
                              ) -> Union[float, str]:
        """Concordance between two embeddings.
        
        This is a wrapper function to implement two embedding concordance metrics based on
        named clusters: EMD and Cluster Distance. When two embeddings can be reasonably
        aligned based on clusters or manual labels, these two metrics calculate the relationships
        between clusters and their distances between two embeddings.
        
        For EMD, the metric considers matched pairs of clusters in both embeddings: for each pair
        in each embedding, the distances between each centroid and all points in the other cluster
        are calculated. The EMD between these two vectors from two embeddings are calculated and
        then averaged across all pairs. 
        
        For Cluster Distance, pairwise rank distance between all cluster centroids are calculated
        in each embedding. Then, the Euclidean distance between these two vectors are taken.
        
        Args:
            embedding (np.ndarray): The first (main) embedding.
            labels_embedding (np.ndarray): Labels for all obervations in the embedding.
            comparison_file (np.ndarray): The second embedding.
            comparison_labels (np.ndarray): The labels for all observations in the comparison embedding.
            comparison_classes (np.ndarray): Which classes in labels to compare. If ``None``, all overlapping labels used.
            method (str): "EMD" or "cluster_distance"
            
        Returns: 
            Union[float, str]: The score or "NA"
            
        Note:
            When there is no overlapping labels, "NA" is automatically returned as ``str``.
        """
        
        if not isinstance(comparison_file, list):
            comparison_file = [comparison_file]
        if not isinstance(comparison_labels, list):
            comparison_labels = [comparison_labels]
        if not isinstance(comparison_classes, list) and comparison_classes is not None:
            comparison_classes = [comparison_classes]
            
        scores: "np.ndarray" = np.zeros(len(comparison_file))
        i: int
        comparison_labels_index: int
        comparison_status: bool = False
        
        for i in range(len(comparison_file)):
            comparison_labels_index = i if len(comparison_labels) == len(comparison_file) else 0
            common_types: "np.ndarray" = np.intersect1d(np.unique(labels_embedding), np.unique(comparison_labels[comparison_labels_index]))
            
            if comparison_classes is not None:
                common_types = np.intersect1d(common_types, comparison_classes)     
            if common_types.shape[0] < 2:
                continue
            
            if method == "emd":
                scores[i] = Metric._concordance_emd(embedding,
                                                    labels_embedding,
                                                    comparison_file[i],
                                                    comparison_labels[comparison_labels_index],
                                                    common_types)
            elif method == "cluster_distance":
                scores[i] = Metric._concordance_cluster_distance(embedding,
                                                                 labels_embedding,
                                                                 comparison_file[i],
                                                                 comparison_labels[comparison_labels_index],
                                                                 common_types)
            comparison_status = True
            
        if comparison_status:
            return np.mean(scores)
        else:
            return "NA"
            
            
    @staticmethod
    def _concordance_emd(embedding: "np.ndarray",
                         labels_embedding: "np.ndarray",
                         comparison_file: "np.ndarray",
                         comparison_labels: "np.ndarray",
                         common_types: "np.ndarray") -> float:
        
        """Embedding concordance EMD.
        
        This is a private that computes the concordance EMD as described in the ``embedding_concordance`` method.
        
        Args:
            embedding (np.ndarray): The first (main) embedding.
            labels_embedding (np.ndarray): Labels for all obervations in the embedding.
            comparison_file (np.ndarray): The second embedding.
            comparison_labels (np.ndarray): The labels for all observations in the comparison embedding.
            common_types (np.ndarray): The common cluster labels for EMD computation.
            
        Returns: 
            float: The EMD score
        """
        
        combinations: List[Tuple[int, ...]] = list(itertools.permutations(common_types, 2))
        embedding_scores: List[float] = []
        comparison_scores: List[float] = []
        
        comb: Tuple[int, ...]
        for comb in combinations:
            indicies_embedding: "np.ndarray" = np.where(labels_embedding == comb[0])[0]
            centroid: "np.ndarray" = np.mean(embedding[indicies_embedding, ], axis=0)
            pwd: "np.ndarray" = scipy.stats.rankdata(np.sqrt(np.sum((centroid - embedding)**2, axis=1)))
            pwd = pwd[np.where(labels_embedding == comb[1])]/(embedding.shape[0])
            embedding_scores.extend(d for d in pwd)
            
            indicies_comparison: "np.ndarray" = np.where(comparison_labels == comb[0])[0]
            centroid_c: "np.ndarray" = np.mean(comparison_file[indicies_comparison,], axis=0)
            pwd_c: "np.ndarray" = scipy.stats.rankdata(np.sqrt(np.sum((centroid_c - comparison_file)**2, axis=1)))
            pwd_c = pwd_c[np.where(comparison_labels == comb[1])]/(comparison_file.shape[0])
            comparison_scores.extend([d for d in pwd_c])
            
        return scipy.stats.wasserstein_distance(embedding_scores, comparison_scores)
    
    
    @staticmethod
    def _concordance_cluster_distance(embedding: "np.ndarray",
                                      labels_embedding: "np.ndarray",
                                      comparison_file: "np.ndarray",
                                      comparison_labels: "np.ndarray",
                                      common_types: "np.ndarray") -> float:
        """Embedding concordance Cluster Distance.
        
        This is a private that computes the concordance Ckuster Distance metric as described in the
        ``embedding_concordance`` method.
        
        Args:
            embedding (np.ndarray): The first (main) embedding.
            labels_embedding (np.ndarray): Labels for all obervations in the embedding.
            comparison_file (np.ndarray): The second embedding.
            comparison_labels (np.ndarray): The labels for all observations in the comparison embedding.
            common_types (np.ndarray): The common cluster labels for EMD computation.
            
        Returns: 
            float: The EMD score
        """
        
        combinations: List[Tuple[int, ...]] = list(itertools.permutations(common_types, 2))
        embedding_scores: "np.ndarray" = np.zeros(len(combinations))
        comparison_scores: "np.ndarray" = np.zeros(len(combinations))
        
        embedding_clusters: "np.ndarray" = np.unique(labels_embedding)
        embedding_centroid: "np.ndarray" = np.zeros((embedding_clusters.shape[0], embedding.shape[1]))
        for i, c in enumerate(embedding_clusters):
            embedding_centroid[i] = np.mean(embedding[np.where(labels_embedding==c)[0],], axis=0)
                
        comparison_clusters: "np.ndarray" = np.unique(comparison_labels)
        comparison_centroid: "np.ndarray" = np.zeros((comparison_clusters.shape[0], comparison_file.shape[1]))
        for i, c in enumerate(comparison_clusters):
            comparison_centroid[i] = np.mean(comparison_file[np.where(comparison_labels==c)[0],], axis=0)

        comb: Tuple[int, ...]
        i: int
        for i, comb in enumerate(combinations):
            embedding_index_0: int = np.where(embedding_clusters == comb[0])[0][0]
            embedding_index_1: int = np.where(embedding_clusters == comb[1])[0][0]
            embedding_cluster_0: "np.ndarray" = embedding_centroid[embedding_index_0]
            embedding_pwd: "np.ndarray" = scipy.stats.rankdata(np.sqrt(np.sum((embedding_cluster_0 - embedding_centroid)**2, axis=1)))
            embedding_scores[i] = embedding_pwd[embedding_index_1,]/embedding_clusters.shape[0]
            
            comparison_index_0: int = np.where(comparison_clusters == comb[0])[0][0]
            comparison_index_1: int = np.where(comparison_clusters == comb[1])[0][0]
            comparison_cluster_0: "np.ndarray" = comparison_centroid[comparison_index_0]
            comparison_pwd: "np.ndarray" = scipy.stats.rankdata(np.sqrt(np.sum((comparison_cluster_0 - comparison_centroid)**2, axis=1)))
            comparison_scores[i] = comparison_pwd[comparison_index_1,]/comparison_clusters.shape[0]
            
        return np.mean(np.abs(embedding_scores-comparison_scores))  
    
    
class PointClusterDistance():
    """Point Cluster Distance
    
    This class is used to compute the Point Cluster Distance. Instead of full pairwise
    distance, this distance metric computes the distance between each cluster centroid
    and all other point. The memory complexity is N_cluster*N instead of (N^2)/2.
    
    Args:
        X (np.ndarray): The input data array.
        labels (np.ndarray): Labels for the data array.
        dist_metric (str): The distance metric to use. This supports "euclidean", "manhattan",
            or "cosine".
        
    Attributes:
        X (np.ndarray): The input data array.
        labels (np.ndarray): Labels for the data array.
        dist_metric (str): The distance metric to use. This supports "euclidean", "manhattan",
            or "cosine".
        dist (np.ndarray, optional): The calculated distance array. The first axis corresponds
            to each observation in the original array and the second axis is all the cluster
            centroids.
    """
    
    def __init__(self, X: "np.ndarray", labels: "np.ndarray", dist_metric: str="euclidean"):
        self.X: "np.ndarray" = X
        self.labels: "np.ndarray" = labels
        self.dist_metric = dist_metric
        self.dist: Optional["np.ndarray"] = None
        
        
    def fit(self, flatten: bool=False) -> Tuple["np.ndarray", "np.ndarray"]:
        """Fit the distance metric.

        This method calculates the distance metric based on the class attributes.
        
        Args:
            flatten: Whether to flatten the return into a 1-d vector

        Returns:
            np.ndarray: The calculate distance array.
            np.ndarray: The unique labels of the input data.
        """
        index: "np.ndarray"
        inverse: "np.ndarray"
        index, inverse = np.unique(self.labels, return_inverse=True)
        
        self.dist = np.empty((self.X.shape[0], index.size))
        centroid: "np.ndarray" = self._cluster_centroid(self.X, index, inverse)
        dist = self._distance_func(dist_metric=self.dist_metric)
        
        i: int
        obs: "np.ndarray"
        for i, obs in enumerate(self.X):
            self.dist[i] = dist(obs, centroid)
            
        if flatten:
            return self.flatten(self.dist), index
        else:
            return self.dist, index
        
    
    @staticmethod
    def flatten(dist: "np.ndarray") -> np.ndarray:
        """Flatten an array

        This method is a wrapper for the ``flatten`` method in ``numpy``.

        Args:
            dist (np.ndarray): The distance array.

        Returns:
            np.ndarray: The flattened array.
        """
        return dist.flatten()
    
    
    @staticmethod
    def _distance_func(dist_metric: str="euclidean") -> Callable[[np.ndarray, np.ndarray], float]:
        if dist_metric == "manhattan":
            return PointClusterDistance._manhattan
        elif dist_metric == "cosine":
            return PointClusterDistance._cosine
        else:    
            return PointClusterDistance._euclidean
    
    
    @staticmethod
    def _euclidean(X1: np.ndarray, X2: np.ndarray) -> float:
        """Euclidean distance

        This is an implementation of the Euclidean distance.

        Args:
            X1 (np.ndarray): The array to calculate Euclidean distance.
            X2 (np.ndarray): The array to calculate Euclidean distance.

        Returns:
            float: The euclidean distance
        """
        return np.sqrt(np.sum(np.square(X1-X2), axis=1))
    
    
    @staticmethod
    def _manhattan(X1: np.ndarray, X2: np.ndarray) -> float:
        """Euclidean distance

        This is an implementation of the Manhattan distance.

        Args:
            X1 (np.ndarray): The array to calculate Manhattan distance.
            X2 (np.ndarray): The array to calculate Manhattan distance.

        Returns:
            float: The Manhattan distance
        """
        return np.sum(np.abs(X1-X2), axis=1)
    
    
    @staticmethod
    def _cosine(X1: np.ndarray, X2: np.ndarray) -> float:
        """Cosine distance

        This is an implementation of the Cosine distance.

        Args:
            X1 (np.ndarray): The array to calculate Cosine distance.
            X2 (np.ndarray): The array to calculate Cosine distance.

        Returns:
            float: The Cosine distance
        """
        return np.sum(X1*X2, axis=1)/(np.sqrt(np.sum(X1**2))*np.sqrt(np.sum(X2**2, axis=1)))
    
    
    @staticmethod
    def _cluster_centroid(X: np.ndarray, index: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        """Find the centroids of clustered data.

        With the input array and labels, this method computes all centroids using
        the labels as clusters.

        Args:
            X (np.ndarray): [description]
            index (np.ndarray): [description]
            inverse (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        centroid: "np.ndarray" = np.empty((index.size, X.shape[1]))
        for i in range(len(index)):
            centroid[i] = np.mean(X[inverse==i])
            
        return centroid