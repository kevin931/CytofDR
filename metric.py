import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import scipy

from typing import Optional, Any, Union, List

class Metric():
    
    @classmethod
    def run_metrics_downsample(cls,
                               data: "np.ndarray",
                               embedding: Union["np.ndarray", List["np.ndarray"]],
                               downsample: int,
                               n_fold: int=1,
                               methods: Optional[Union[str, List[str]]]="all",
                               labels: Optional["np.ndarray"]=None,
                               embedding_names: Optional["np.ndarray"]=None,
                               k: int=5
                               ) -> List[List[Union[str, float]]]:
        
        
        if not isinstance(embedding, list):
            embedding = [embedding]
        
        data_downsample: "np.ndarray"
        embedding_downsample: List["np.ndarray"]
        labels_downsample: Optional["np.ndarray"]=None
        results: List[List[Union[str, float]]] = []
        results_numeric: List[List[Any]] = []
        
        e: "np.ndarray"
        
        for _ in range(n_fold):
            index: Union["np.ndarray", int] = np.random.randint(0, data.shape[0], size=downsample)
            
            data_downsample = data[index, :]
            embedding_downsample = [e[index, :] for e in embedding]
            
            if labels is not None:
                labels_downsample = labels[index]
            
            results = cls.run_metrics(data=data_downsample,
                                      embedding=embedding_downsample,
                                      methods=methods,
                                      labels=labels_downsample,
                                      embedding_names=embedding_names,
                                      k=k)
            
            results_numeric.append(results[1])
        
        results_numeric = list(np.average(results_numeric, axis=0))
        results[1] = results_numeric #type: ignore
        
        return results
        
    
    @classmethod
    def run_metrics(cls,
                    data: "np.ndarray",
                    embedding: Union["np.ndarray", List["np.ndarray"]],
                    methods: Optional[Union[str, List[str]]]="all",
                    labels: Optional["np.ndarray"]=None,
                    embedding_names: Optional["np.ndarray"]=None,
                    k: int=5
                    ) -> List[List[Union[str, float]]]:
            
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
                methods.extend(["npe", "random_forest"])
            
        if any(m in methods for m in ["npe"]) and labels is None: 
            raise ValueError("'labels' must be provided for NPE and KNC.")
        
        if any(m in methods for m in ["knn", "npe", "neighborhood_agreement", "neighborhood_trustworthiness"]) and not isinstance(k, int): 
            raise TypeError("'k' must be an integer for NPE, KNN, Neighborhood Agreement, or Neighborhood Trustworthiness.")
        
        e: "np.ndarray"
        i: int
        
        if any(m in methods for m in ["pearsonr", "spearmanr", "residual_variance", "neighborhood_trustworthiness", "emd"]): 
            data_pairwise_distance: "np.ndarray" = cls._pairwise_distance(data=data, metric="euclidean")
            embedding_pairwise_distance: List["np.ndarray"] = []
            for e in embedding:
                embedding_pairwise_distance.append(cls._pairwise_distance(data=e, metric="euclidean"))
            
        if any(m in methods for m in ["knn", "npe", "neighborhood_agreement", "neighborhood_trustworthiness"]):
            knn_model_data: "NearestNeighbors" = NearestNeighbors(n_neighbors=k).fit(data)
            knn_model_embedding: List["NearestNeighbors"] = []
            for e in embedding:
                knn_model_embedding.append(NearestNeighbors(n_neighbors=k).fit(e))
            
        results: List[List[Any]] = [[],[], []]
        
        for i, e in enumerate(embedding):            
        
            if "pearsonr" in methods:
                cor: float=cls.correlation(x=data_pairwise_distance, y=embedding_pairwise_distance[i], metric="Pearson")
                results[0].append("correlation_pearson")
                results[1].append(cor)
                results[2].append(embedding_names[i])
                
            if "spearmanr" in methods:
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
                results[0].append("random_forest")
                results[1].append(cls.random_forest(embedding=e, labels=labels))
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
        
        if r is None:
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
        
        data_neighbors: "np.ndarray"
        embedding_neighbors: "np.ndarray"
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            data_neighbors = NearestNeighbors(n_neighbors=k).fit(data).kneighbors(return_distance=False) #type:ignore
        else:
            data_neighbors = knn_model_data.kneighbors(return_distance=False) #type:ignore
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            embedding_neighbors = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(return_distance=False) #type:ignore
        else:
            embedding_neighbors = knn_model_embedding.kneighbors(return_distance=False) #type:ignore
        
        i: int
        intersect: int = 0
        for i in range(data_neighbors.shape[0]):
            intersect += np.intersect1d(data_neighbors[i], embedding_neighbors[i]).shape[0]
        
        return intersect/data_neighbors.size

    
    @staticmethod
    def NPE(data: "np.ndarray",
            embedding: "np.ndarray",
            labels: "np.ndarray",
            k: Optional[int]=None,
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
            agreement (float): Neighborhood agreement.
        
        '''
        
        if knn_model_data is None and k is None:
            raise ValueError("'k' is required if 'knn_model_data' not supplied")
        elif knn_model_data is None:
            knn_model_data = NearestNeighbors(n_neighbors=k).fit(data).kneighbors(return_distance=False)
        else:
            knn_model_data.kneighbors(return_distance=False)
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            knn_model_embedding = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(return_distance=False)
        else:
            knn_model_embedding.kneighbors(return_distance=False)
        
        classes, classes_index = np.unique(labels, return_inverse=True)
        
        same_class_data: "np.ndarray" = np.zeros(data.shape(0))
        same_class_embedding: "np.ndarray" = np.zeros(data.shape(0)) 
        i: int 
        for  i in range(data.shape[0]):
            i_index = classes_index[i]
            i_neighbors_data = knn_model_data[i]
            i_neighbors_embedding = knn_model_embedding[i]
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
                               k: Optional[int]=None,
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
            knn_model_data = NearestNeighbors(n_neighbors=k).fit(data).kneighbors(return_distance=False)
        else:
            knn_model_data = knn_model_data.kneighbors(return_distance=False)
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_embedding' not supplied")
        elif knn_model_embedding is None:
            knn_model_embedding = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(return_distance=False)
        else:
            knn_model_embedding = knn_model_embedding.kneighbors(return_distance=False)
        
        i: int
        agreement: float = 0.0
        for i in range(data.shape[0]):
            agreement += np.intersect1d(knn_model_data[i], knn_model_embedding[i]).shape[0]
        
        agreement = (agreement/(k*data.shape[0])*(data.shape[0]-1)-k)/(data.shape[0]-1-k)
        
        return agreement
    
    
    @staticmethod
    def neighborhood_trustworthiness(data: "np.ndarray",
                                     embedding: "np.ndarray",
                                     dist_data: Optional["np.ndarray"]=None,
                                     k: int=None,
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
            knn_model_data = NearestNeighbors(n_neighbors=k).fit(data).kneighbors(return_distance=False)
        else:
            knn_model_data = knn_model_data.kneighbors(return_distance=False)
            
        if knn_model_embedding is None and k is None:
            raise ValueError("'k' is required if 'knn_model_y' not supplied")
        elif knn_model_embedding is None:
            knn_model_embedding = NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(return_distance=False)
        else:
            knn_model_embedding = knn_model_embedding.kneighbors(return_distance=False)

        if dist_data is None:
            dist_data = scipy.spatial.distance.squareform(Metric._pairwise_distance(data, metric="euclidean"))
        else:
            dist_data = scipy.spatial.distance.squareform(dist_data)
            
        dist_data = scipy.stats.rankdata(dist_data, axis=1)
        
        score: float = 0
        i: int
        for i in range(data.shape[0]):
            neighbor_diff: "np.ndarray" = np.setdiff1d(knn_model_embedding[i], knn_model_data[i], assume_unique=True)
            score += np.sum(dist_data[i, neighbor_diff] - k)
            
        score = 1 - 2*score/(data.shape[0]*k*(2*data.shape[0]-3*k-1))
        
        return score
        
    
    @staticmethod
    def EMD(data: "np.ndarray",
            embedding: "np.ndarray", 
            dist_data: Optional["np.ndarray"]=None,
            dist_embedding: Optional["np.ndarray"]=None) -> float:
        
        if dist_data is None:
            dist_data = Metric._pairwise_distance(data, metric="euclidean")
            
        if dist_embedding is None:
            dist_embedding = Metric._pairwise_distance(embedding, metric="euclidean")
        
        return scipy.stats.wasserstein_distance(dist_data, dist_embedding)
    
    
    @staticmethod
    def random_forest(embedding: "np.ndarray",
                      labels: "np.ndarray") -> float:
        
        embedding_train, embedding_test, labels_train, labels_test = train_test_split(embedding, labels, test_size=0.33)

        rf: "RandomForestClassifier" = RandomForestClassifier().fit(embedding_train, labels_train)
        predictions: "np.ndarray" = rf.predict(embedding_test)
        
        return float(np.mean(np.equal(predictions, labels_test)))


if __name__ == "__main__":
    a = np.random.rand(100, 5)
    b = np.random.rand(100, 2)
    
    results = Metric.run_metrics_downsample(a,
                                            b, 
                                            50, 
                                            n_fold=3, 
                                            methods=["pearsonr", "knn"])
    print(results)