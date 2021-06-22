import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import scipy.spatial
import scipy.stats
import sklearn.metrics

from fileio import FileIO

import copy
from typing import Optional, Any, Union, List

class Metric():
    
    @classmethod
    def run_metrics_downsample(cls,
                               data: "np.ndarray",
                               embedding: Optional[Union["np.ndarray", List["np.ndarray"]]]=None,
                               downsample: Optional[int]=None,
                               downsample_indices: Optional[List["np.ndarray"]]=None,
                               n_fold: int=1,
                               methods: Union[str, List[str]]="all",
                               labels: Optional["np.ndarray"]=None,
                               labels_embedding: Optional["np.ndarray"]=None,
                               embedding_names: Optional["np.ndarray"]=None,
                               k: int=5,
                               save_indices_dir: Optional[str]=None
                               ) -> Optional[List[List[Union[str, float]]]]:
        
        '''Run methods with downsampling.
        
        This method first downsamples and runs the methods with the ``run_metrics`` method. This
        is done for efficiency reasons as the current pairwise distance metric is can be memory
        intensive with large datasets.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            downsample (int): The sample size of downsampling.
            n_fold (int): Downsample n times and average the results.
            methods (Union[str, List[str]]): The metrics to run.
            labels ("np.ndarray", optional): True labels or labels from the original space.
            labels_embedding ("np.ndarray", optional): Classification or clustering labels from the embedding space.
            embedding_names ("np.ndarray", optional): Names of the embedding methods to be saved.
            k (int): The number of neighbors for KNN.

        Returns:
            List[List[Union[str, float]]]: A nested list of results with names of metrics, metrics results,
                name of embedding, and downsample index.
        '''
        
        if downsample is None and downsample_indices is None:
            raise ValueError("Either 'downsample' or 'downsample_indices' must be provided.")
        
        if downsample_indices is not None:
            n_fold = len(downsample_indices)
        
        if not isinstance(embedding, list) and embedding is not None:
            embedding = [embedding]
        
        data_downsample: "np.ndarray"
        embedding_downsample: List["np.ndarray"]
        labels_downsample: Optional["np.ndarray"]=None
        labels_embedding_downsample: Optional["np.ndarray"]=None
        results: List[List[Union[str, float]]] = []
        results_downsample_index: List[int] = []
        results_combined: List[List[Any]] = [[],[],[]]
        
        n: int 
        for n in range(n_fold):
            
            index: "np.ndarray"
            if downsample_indices is None:
                index = np.random.choice(data.shape[0], size=downsample)
            else:
                index = downsample_indices[n].astype(int)
                  
            if save_indices_dir is not None:
                file_name: str = "index_{}".format(n)
                FileIO.save_np_array(index, save_indices_dir, file_name=file_name)
                
            if embedding is None:
                continue
                
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
                                      k=k)
            
            for col in range(len(results)):
                results_combined[col].extend(results[col])
                
            results_downsample_index.extend([n]*len(results[0]))
        
        if embedding is None:
            return None
        else:
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
            
        if "all" in methods:
            methods = ["pearsonr", "spearmanr", "residual_variance", "knn", "neighborhood_agreement", "neighborhood_trustworthiness", "emd"]
            if labels is not None:
                methods.extend(["npe", "random_forest", "silhouette"])
            if labels is not None and labels_embedding is not None:
                methods.extend(["ari", "mni"])
            print(methods)
            
        if any(m in methods for m in ["npe", "random_forest", "silhouette", "ari", "mni"]) and labels is None: 
            raise ValueError("'labels' must be provided for NPE, random forest, silhouette, ARI, and MNI.")
        
        if any(m in methods for m in ["ari", "nmi"]) and labels_embedding is None: 
            raise ValueError("'labels_embedding' must be provided for ARI, and MNI.")
        
        if any(m in methods for m in ["knn", "npe", "neighborhood_agreement", "neighborhood_trustworthiness"]) and not isinstance(k, int): 
            raise TypeError("'k' must be an integer for NPE, KNN, Neighborhood Agreement, or Neighborhood Trustworthiness.")
        
        e: "np.ndarray"
        i: int
        
        data_pairwise_distance: Optional["np.ndarray"] = None
        embedding_pairwise_distance: Optional[List["np.ndarray"]] = None
        
        if any(m in methods for m in ["pearsonr", "spearmanr", "residual_variance", "neighborhood_trustworthiness", "emd"]): 
            data_pairwise_distance = cls._pairwise_distance(data=data, metric="euclidean")
            embedding_pairwise_distance = []
            for e in embedding:
                embedding_pairwise_distance.append(cls._pairwise_distance(data=e, metric="euclidean"))
        
        knn_model_data: Optional["NearestNeighbors"] = None
        knn_model_embedding: Optional[List["NearestNeighbors"]] = None
        
        if any(m in methods for m in ["knn", "npe", "neighborhood_agreement", "neighborhood_trustworthiness"]):
            knn_model_data = NearestNeighbors(n_neighbors=k).fit(data)
            knn_model_embedding = []
            for e in embedding:
                knn_model_embedding.append(NearestNeighbors(n_neighbors=k).fit(e))
            
        results: List[List[Any]] = [[],[], []]
        
        for i, e in enumerate(embedding):
        
            if "pearsonr" in methods:
                assert data_pairwise_distance is not None and embedding_pairwise_distance is not None
                cor: float=cls.correlation(x=data_pairwise_distance, y=embedding_pairwise_distance[i], metric="Pearson")
                results[0].append("correlation_pearson")
                results[1].append(cor)
                results[2].append(embedding_names[i])
                
            if "spearmanr" in methods:
                assert data_pairwise_distance is not None and embedding_pairwise_distance is not None
                cor: float=cls.correlation(x=data_pairwise_distance, y=embedding_pairwise_distance[i], metric="Spearman")
                results[0].append("correlation_spearman")
                results[1].append(cor)
                results[2].append(embedding_names[i])
                
            if "residual_variance" in methods:
                # TODO: Implement r when pearsonr is not None.
                results[0].append("residual_variance")
                results[1].append(cls.residual_variance(x=data_pairwise_distance, y=embedding_pairwise_distance[i]))
                results[2].append(embedding_names[i])
                
            if "knn" in methods:
                results[0].append("knn")
                results[1].append(cls.KNN(data=data, embedding=e, knn_model_data=knn_model_data, knn_model_embedding=knn_model_embedding[i], k=k))
                results[2].append(embedding_names[i])
            
            if "npe" in methods:
                assert labels is not None
                results[0].append("npe")
                results[1].append(cls.NPE(data=data, embedding=e, labels=labels, k=k, knn_model_data=knn_model_data, knn_model_embedding=knn_model_embedding[i]))
                results[2].append(embedding_names[i])
                
            if "neighborhood_agreement" in methods:
                results[0].append("neighborhood_agreement")
                results[1].append(cls.neighborhood_agreement(data=data, embedding=e, k=k, knn_model_data=knn_model_data, knn_model_embedding=knn_model_embedding[i]))
                results[2].append(embedding_names[i])
            
            if "neighborhood_trustworthiness" in methods:
                results[0].append("neighborhood_trustworthiness")
                results[1].append(cls.neighborhood_trustworthiness(data=data, embedding=e, dist_data=data_pairwise_distance, k=k, knn_model_data=knn_model_data, knn_model_embedding=knn_model_embedding[i]))
                results[2].append(embedding_names[i])
                
            if "emd" in methods:
                results[0].append("emd")
                results[1].append(cls.EMD(data=data, embedding=e, dist_data=data_pairwise_distance, dist_embedding=embedding_pairwise_distance[i]))
                results[2].append(embedding_names[i])
                
            if "random_forest" in methods:
                assert labels is not None
                results[0].append("random_forest")
                results[1].append(cls.random_forest(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "silhouette" in methods:
                assert labels is not None
                results[0].append("silhouette")
                results[1].append(cls.silhouette(embedding=e, labels=labels))
                results[2].append(embedding_names[i])
                
            if "nmi" in methods:
                assert labels is not None and labels_embedding is not None
                results[0].append("nmi")
                results[1].append(cls.NMI(labels=labels, labels_embedding=labels_embedding))
                results[2].append(embedding_names[i])
                
            if "ari" in methods:
                assert labels is not None and labels_embedding is not None
                results[0].append("ari")
                results[1].append(cls.ARI(labels=labels, labels_embedding=labels_embedding))
                results[2].append(embedding_names[i])
            
        return results
        
    
    @staticmethod
    def _pairwise_distance(data: "np.ndarray", metric:str) -> "np.ndarray":
        return scipy.spatial.distance.pdist(data, metric=metric)
    
    
    @staticmethod
    def _KNN(x: "np.ndarray", k: int) -> "NearestNeighbors":
        return NearestNeighbors(n_neighbors=k).fit(x)
    
    
    @staticmethod
    def correlation(x: "np.ndarray",
                    y: "np.ndarray",
                    metric: str="Pearson") -> float:
        
        '''Correlation
        
        This method computes the pearson or spearman correlation between the inputs.
        
        Parameters:
            x (np.ndarray): The first dimension. 
            y (np.ndarray): The second dimension.
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
            x (np.ndarray, optional): The first dimension. 
            y (np.ndarray, optional): The second dimension.
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
    def KNN(data: "np.ndarray",
            embedding: "np.ndarray",
            k: int=None,
            knn_model_data: "NearestNeighbors"=None,
            knn_model_embedding: "NearestNeighbors"=None) -> float:
        
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
        
        data_neighbors: "np.ndarray"
        embedding_neighbors: "np.ndarray"
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            data_neighbors = NearestNeighbors(n_neighbors=k).fit(data).kneighbors()[1]
        else:
            data_neighbors = knn_model_data.kneighbors()[1]
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            embedding_neighbors = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors()[1]
        else:
            embedding_neighbors = knn_model_embedding.kneighbors()[1] 
        
        i: int
        intersect: int = 0
        for i in range(data_neighbors.shape[0]):
            intersect += np.intersect1d(data_neighbors[i], embedding_neighbors[i]).shape[0]
        
        return intersect/data_neighbors.size

    
    @staticmethod
    def NPE(data: "np.ndarray",
            embedding: "np.ndarray",
            labels: "np.ndarray",
            k: int=5,
            knn_model_data: Optional["NearestNeighbors"]=None,
            knn_model_embedding: Optional["NearestNeighbors"]=None) -> float:
        
        '''Neighborhood Proportion Error (NPE)
        
        The NPE metric is proposed by Konstorum et al. (2019). It measures the total variation distance between
        the proportion of nearest points belonging to the same class of each point in the HD and LD space. The
        lower the NPE, the more similar the embedding and the original data are.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.
            k (int, optional): The number of neighbors for KNN. This is required when either knn_model_data
                or knn_model_embedding is not supplied.
            knn_model_data (sklearn.neighbors.NearestNeighbors, optional):
                A fitted instance of ``sklearn.neighbors.NearestNeighbors`` with ``data``.
            knn_model_embedding (sklearn.neighbors.NearestNeighbors, optional): 
                A fitted instance of ``sklearn.neighbors.NearestNeighbors`` with ``embedding``.

        Returns:
            float: Neighborhood proportion error.
        '''
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            data_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(data).kneighbors()[1]
        else:
            data_neighbors: "np.ndarray" = knn_model_data.kneighbors()[1]
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            embedding_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors()[1]
        else:
            embedding_neighbors: "np.ndarray" = knn_model_embedding.kneighbors()[1]
        
        classes: "np.ndarray"
        classes_index: "np.ndarray"
        classes, classes_index = np.unique(labels, return_inverse=True)
        
        same_class_data: "np.ndarray" = np.zeros(data.shape[0])
        same_class_embedding: "np.ndarray" = np.zeros(data.shape[0]) 
        
        i: int 
        for  i in range(data.shape[0]):
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
    def neighborhood_agreement(data: "np.ndarray",
                               embedding: "np.ndarray",
                               k: int=5,
                               knn_model_data: Optional["NearestNeighbors"]=None,
                               knn_model_embedding: Optional["NearestNeighbors"]=None) -> float:
        '''Neighborhood Agreement
        
        The Neighborhood Agreement metric is proposed by Lee et al. (2015). It measures
        the intersection of k-nearest neighbors (KNN) of each point in HD and LD space. The
        result is subsequently rescaled to measure the improvement over a random embedding.
        This measure is conceptually similar to ``Metric.KNN`` such that they both measure
        the agreement of KNN, but ``Metric.KNN`` simply takes the average of the KNN graph
        agreement without any scaling.
        
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
            agreement (float): Neighborhood agreement.
        
        '''
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            data_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(data).kneighbors()[1]
        else:
            data_neighbors: "np.ndarray" = knn_model_data.kneighbors()[1]
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            embedding_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors()[1]
        else:
            embedding_neighbors: "np.ndarray" = knn_model_embedding.kneighbors()[1]
        
        i: int
        agreement: float = 0.0
        for i in range(data.shape[0]):
            agreement += np.intersect1d(data_neighbors[i], embedding_neighbors[i]).shape[0]
        
        agreement = (agreement/(k*data.shape[0])*(data.shape[0]-1)-k)/(data.shape[0]-1-k)
        
        return agreement
    
    
    @staticmethod
    def neighborhood_trustworthiness(data: "np.ndarray",
                                     embedding: "np.ndarray",
                                     dist_data: Optional["np.ndarray"]=None,
                                     k: int=5,
                                     knn_model_data: "NearestNeighbors"=None,
                                     knn_model_embedding: "NearestNeighbors"=None) -> float:
        '''Neighborhood Trustworthiness 
        
        The Neighborhood Truestworthiness is proposed by Venna and Kaski (2001). It measures
        trustworthiness by measuring the ranked distane of new points entering the defined
        neighborhood size in the embedding. The higher the new points are ranked based on the
        original HD space distance matrix, the less trustworthy the new embedding is. The measure
        is scaled between 0 and 1 with a higher score reflecting a more trustworthy embedding.
        
        Parameters:
            data (np.ndarray): The input high-dimensional array.
            embedding (np.ndarray): The low-dimensional embedding.

        Returns:
            agreement (float): Neighborhood agreement.
        
        '''
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            data_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(data).kneighbors()[1]
        else:
            data_neighbors: "np.ndarray" = knn_model_data.kneighbors()[1]
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_y' not supplied")
        elif knn_model_embedding is None:
            embedding_neighbors: "np.ndarray" = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors()[1]
        else:
            embedding_neighbors: "np.ndarray" = knn_model_embedding.kneighbors()[1]

        if dist_data is None:
            dist_data = scipy.spatial.distance.squareform(Metric._pairwise_distance(data, metric="euclidean"))
        else:
            dist_data = scipy.spatial.distance.squareform(dist_data)
            
        dist_data = scipy.stats.rankdata(dist_data, axis=1)
        
        score: float = 0
        i: int
        for i in range(data.shape[0]):
            neighbor_diff: "np.ndarray" = np.setdiff1d(embedding_neighbors[i], data_neighbors[i], assume_unique=True)
            score += np.sum(dist_data[i, neighbor_diff] - k)
            
        score = 1 - 2*score/(data.shape[0]*k*(2*data.shape[0]-3*k-1))
        
        return score
        
    
    @staticmethod
    def EMD(data: "np.ndarray",
            embedding: "np.ndarray", 
            dist_data: Optional["np.ndarray"]=None,
            dist_embedding: Optional["np.ndarray"]=None) -> float:
        
        '''Earth Mover's Distance (EMD)
        
        This metric computes the EMD between the pairwise distance of between points in the
        high and low dimensional space. This implementation uses the ``scipy.stats.wasserstein_distance``.
        The usage of EMD is proposed in Heiser & Lou (2020).
        
        Parameters:
            embedding (np.ndarray): The low-dimensional embedding.
            labels (np.ndarray): The class labels of each observation.

        Returns:
            float: Earth mover's distance.
        
        '''
        
        if dist_data is None:
            dist_data = Metric._pairwise_distance(data, metric="euclidean")
            
        if dist_embedding is None:
            dist_embedding = Metric._pairwise_distance(embedding, metric="euclidean")
        
        return scipy.stats.wasserstein_distance(dist_data, dist_embedding)
    
    
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
        '''
        
        return sklearn.metrics.adjusted_rand_score(labels, labels_embedding)