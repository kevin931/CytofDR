import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from annoy import AnnoyIndex

import scipy.spatial
import scipy.stats
import sklearn.metrics

from annoy import AnnoyIndex
import itertools
from typing import Optional, Any, Union, List, Tuple, Callable
import warnings


class EvaluationMetrics():
    """Evaluation metrics for dimension reduction
    
    This class contains methods to run evluation metrics.
    """
        
    @staticmethod
    def build_annoy(data: "np.ndarray",
                    saved_annoy_path: Optional[str]=None,
                    k: int=5) -> "np.ndarray":
        '''Build ANNOY and returns nearest neighbors.
        
        This is a utility function for building ANNOY models and returning the nearest-neighbor matrices for original
        space data and low-dimensional embedding. 
        
        
        : param data: The input high-dimensional array.
        : param saved_annoy_path: The path to pre-built ANNOY model for original data.
        : param k: The number of neighbors.

        :return: Nearest-neighbor matrices of original space data.
        '''
        
        if saved_annoy_path is not None:
            annoy_model_data = Annoy.load_annoy(path=saved_annoy_path, ncol=data.shape[1])
        else:
            annoy_model_data = Annoy.build_annoy(data)
            
        annoy_data_neighbors = np.empty((data.shape[0], k), dtype=int)
        for i in range(data.shape[0]):
            data_neighbors: List[int] = annoy_model_data.get_nns_by_item(i, k+1)
            # Try remove the i-th obs itself
            try:
                data_neighbors.remove(i)
            except ValueError: # pragma: no cover
                data_neighbors = data_neighbors[0:(len(data_neighbors)-1)]
                
            annoy_data_neighbors[i] = data_neighbors
        
        return annoy_data_neighbors
        
    
    @staticmethod
    def correlation(x: "np.ndarray",
                    y: "np.ndarray",
                    metric: str="Pearson") -> float:
        '''Calculate Correlation Coefficient
        
        This method computes the pearson or spearman correlation between the inputs.

        :param x: The first 1D array. 
        :param y: The second 1D array.
        :param metric: The metric to use. 'Pearson' or 'Spearman', defaults to "Pearson".

        :return: Correlation coefficient.
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
        
        :param x: The first 1D array, optional. 
        :param y: The second 1D array, optional.
        :param r: Pearson correlation between x and y, optional.

        Returns:
            float: Redisual variance.
        '''
        
        if r is None:
            if x is None or y is None:
                raise ValueError("Either 'r' or both 'x' and 'y' is needed.")
            else:
                cor = EvaluationMetrics.correlation(x, y, metric="Pearson")
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
        
        :param x: The first distribution x as a 1D array.
        :param y: The second distribution y as a 1D array.

        :return: Earth mover's distance.
        '''
        return scipy.stats.wasserstein_distance(x, y)

    
    @staticmethod
    def KNN(data_neighbors: "np.ndarray",
            embedding_neighbors: "np.ndarray") -> float:
        '''K-Nearest Neighbors Preservation (KNN)
        
        The KNN metric computes the percentage of k-neighbors of each point is preserved in the
        embedding space, and it is average across the entire dataset.
        
        .. note:: This method is not used to calculate KNN itself.

        :param data_neighbors: A nearest-neighbor array of the original data.
        :param embedding_neighbors: A nearest-neighbor array of the embedding.

        :return: K-nearest neighbors preservation.
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
        
        :param data_neighbors: A nearest-neighbor array of the original data.
        :param embedding_neighbors: A nearest-neighbor array of the embedding.
        :param labels: The class labels of each observation.

        :return: Neighborhood proportion error.
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
        
        :param data_neighbors: A nearest-neighbor array of the original data.
        :param embedding_neighbors: A nearest-neighbor array of the embedding.

        :return: Neighborhood agreement.
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
        
        :param data_neighbors: A nearest-neighbor matrix of the original data.
        :param embedding_neighbors: A nearest-neighbor matrix of the embedding.
        :param dist_data: A pairwise distance matrix for the original data.

        :return: Neighborhood trustworthiness.
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
        
        :param embedding: The low-dimensional embedding.
        :param labels: The class labels of each observation.

        :return: Random forest prediction accuracy.
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
        
        :param embedding: The low-dimensional embedding.
        :param labels: The class labels of each observation.

        :return: Silhouette score.
        '''
        
        return sklearn.metrics.silhouette_score(embedding, labels)
    
    
    @staticmethod
    def NMI(x_labels: "np.ndarray",
            y_labels: "np.ndarray") -> float:
        '''Normalized Mutual Information (NMI)
        
        The NMI metric computes the mutual information between labels of the original space
        and the embeeding space and then normalizes it with the larger entroy of the two vectors.
        This metric is a measure of clustering performance before and after dimension reduction,
        and it is used in Xiang et al. (2021).
        
        :param x_labels: The first set of labels. 
        :param y_labels: The second set of labels on the same data. 

        :return: Silhouette score.
        '''
        
        mi: float = sklearn.metrics.mutual_info_score(x_labels, y_labels)
        
        labels_count: "np.ndarray"
        embedding_count: "np.ndarray"
        
        _ , labels_count = np.unique(x_labels, return_counts=True)
        _ , embedding_count = np.unique(y_labels, return_counts=True)
        
        labels_count = labels_count/x_labels.size
        embedding_count = embedding_count/y_labels.size
        normalization: float = np.maximum(scipy.stats.entropy(labels_count),
                                          scipy.stats.entropy(embedding_count))
        
        return mi/normalization
    
    
    @staticmethod
    def ARI(x_labels: "np.ndarray",
            y_labels: "np.ndarray") -> float:
        '''Adjusted Rand Index (ARI)
        
        The ARI uses the labels from the original space and the embedding space to measure
        the similarity between them using pairs. It is used in Xiang et al. (2021).
        
        :param x_labels: The first set of labels. 
        :param y_labels: The second set of labels on the same data. 

        :return: ARI.
            
        :References:
        
        This implementation adapts from sklearn's implementation of ARI with a bug fix of overflow
        issue. 
        
        .. code-block:: text
        
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
            
        :License:
        
        .. code-block:: text
        
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
        
        confusion: "np.ndarray" = sklearn.metrics.pair_confusion_matrix(x_labels, y_labels)
        
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
        
        :param embedding: The low-dimensional embedding.
        :param labels: The class labels of each observation.

        :return: Calinski-Harabasz Index.
        '''
        
        return sklearn.metrics.calinski_harabasz_score(embedding, labels)
    
    
    @staticmethod
    def davies_bouldin(embedding: "np.ndarray",
                       labels: "np.ndarray") -> float:
        '''Davies-Bouldin Index
        
        This metric computes the Davies-Bouldin index of clusters in the embedding space. Ideally,
        clusters should be coherent, and using labels obtained from the original space can
        evaluate the effectiveness of the embedding technique.
        
        :param embedding: The low-dimensional embedding.
        :param labels: The class labels of each observation.

        :return: Davies-Bouldin Index.
        '''
        
        return sklearn.metrics.davies_bouldin_score(embedding, labels)
    
    
    @staticmethod
    def embedding_concordance(embedding: "np.ndarray",
                              labels_embedding: "np.ndarray",
                              comparison_file: Union["np.ndarray", List["np.ndarray"]],
                              comparison_labels: Union["np.ndarray", List["np.ndarray"]],
                              comparison_classes: Optional[List[str]]=None,
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
        
        :param embedding: The first (main) embedding.
        :param labels_embedding: Labels for all obervations in the embedding.
        :param comparison_file: The second embedding.
        :param comparison_labels: The labels for all observations in the comparison embedding.
        :param comparison_classes: Which classes in labels to compare. At least two classes need
            to be provided for this to work; otherwise, `NA` will be returned. If ``None``, all
            overlapping labels used, optional
        :param method: "emd" or "cluster_distance", defaults to "emd"
            
        :return: The score or "NA"
            
        .. Note:: When there is no overlapping labels, "NA" is automatically returned as ``str``.
        
        .. deprecated:: 0.2.0
        
            Passing in `str` for the `comparison_classes` parameter is deprecated
            and will be removed in futrue versions.
        """
        
        if not isinstance(comparison_file, list):
            comparison_file = [comparison_file]
        if not isinstance(comparison_labels, list):
            comparison_labels = [comparison_labels]
        if not isinstance(comparison_classes, list) and comparison_classes is not None:
            comparison_classes = [comparison_classes]
            warnings.warn("Passing in a non-list parameter is deprecated. Use a list instead.", DeprecationWarning, stacklevel=2)
            
        method = method.lower()
            
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
                scores[i] = EvaluationMetrics._concordance_emd(embedding,
                                                               labels_embedding,
                                                               comparison_file[i],
                                                               comparison_labels[comparison_labels_index],
                                                               common_types)
            elif method == "cluster_distance":
                scores[i] = EvaluationMetrics._concordance_cluster_distance(embedding,
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
        
        This is a private method that computes the concordance EMD as described in the ``embedding_concordance`` method.
        
        :param embedding: The first (main) embedding.
        :param labels_embedding: Labels for all obervations in the embedding.
        :param comparison_file: The second embedding.
        :param comparison_labels: The labels for all observations in the comparison embedding.
        :param common_types: The common cluster labels for EMD computation.
            
        :return: The EMD score
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
        
        This is a private method that computes the concordance Ckuster Distance metric as described in the
        ``embedding_concordance`` method.
        
        :param embedding: The first (main) embedding.
        :param labels_embedding: Labels for all obervations in the embedding.
        :param comparison_file: The second embedding.
        :param comparison_labels: The labels for all observations in the comparison embedding.
        :param common_types: The common cluster labels for EMD computation.
            
        :return: The EMD score
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
    
    :param X: The input data array.
    :param labels: Labels for the data array.
    :param dist_metric: The distance metric to use. This supports "euclidean", "manhattan", or "cosine", defaults to "euclidean"
        
    :Attributes:
        - **X**: The input data array.
        - **labels**: Labels for the data array.
        - **dist_metric**: The distance metric to use. This supports "euclidean", "manhattan", or "cosine", defaults to "euclidean"
        - **dist**: The calculated distance array. The first axis corresponds to each observation in the original array and the second axis is all the cluster centroids, optional.
    """
    
    def __init__(self, X: "np.ndarray", labels: "np.ndarray", dist_metric: str="euclidean"):
        self.X: "np.ndarray" = X
        self.labels: "np.ndarray" = labels
        self.dist_metric = dist_metric.lower()
        self.dist: Optional["np.ndarray"] = None
        self.index: Optional["np.ndarray"] = None
        
        
    def fit(self, flatten: bool=False) -> "np.ndarray":
        """Fit the distance metric.

        This method calculates the distance metric based on the class attributes.
        
        :param flatten: Whether to flatten the return into a 1-d vector

        :return: The calculate distance array.
        """
        index: "np.ndarray"
        inverse: "np.ndarray"
        index, inverse = np.unique(self.labels, return_inverse=True)
        
        self.dist = np.empty((self.X.shape[0], index.size))
        self.index = index
        centroid: "np.ndarray" = self._cluster_centroid(self.X, index, inverse)
        dist = self._distance_func(dist_metric=self.dist_metric)
        
        i: int
        obs: "np.ndarray"
        for i, obs in enumerate(self.X):
            self.dist[i] = dist(obs, centroid)
            
        if flatten:
            return self.flatten(self.dist)
        else:
            return self.dist
        
    
    @staticmethod
    def flatten(dist: "np.ndarray") -> np.ndarray:
        """Flatten an array

        This method is a wrapper for the ``flatten`` method in ``numpy``.

        :param dist: The distance array.

        :return: The flattened array.
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
        :param X1: The array to calculate Euclidean distance.
        :param X2: The array to calculate Euclidean distance.

        :return: The euclidean distance
        """
        return np.sqrt(np.sum(np.square(X1-X2), axis=1))
    
    
    @staticmethod
    def _manhattan(X1: np.ndarray, X2: np.ndarray) -> float:
        """Euclidean distance

        This is an implementation of the Manhattan distance.

        Args:
        :param X1: The array to calculate Manhattan distance.
        :param X2: The array to calculate Manhattan distance.

        :return: The Manhattan distance
        """
        return np.sum(np.abs(X1-X2), axis=1)
    
    
    @staticmethod
    def _cosine(X1: np.ndarray, X2: np.ndarray) -> float:
        """Cosine distance

        This is an implementation of the Cosine distance.

        Args:
        :param X1: The array to calculate Cosine distance.
        :param X2: The array to calculate Cosine distance.

        :return: The Cosine distance
        """
        return np.sum(X1*X2, axis=1)/(np.sqrt(np.sum(X1**2))*np.sqrt(np.sum(X2**2, axis=1)))
    
    
    @staticmethod
    def _cluster_centroid(X: np.ndarray, index: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        """Find the centroids of clustered data.

        With the input array and labels, this method computes all centroids using
        the labels as clusters.

        :param X: The original data array.
        :param index: The unique IDs of the observations.
        :param inverse: The indicies to reconstruct the original array.

        :return: The cluster centroids.
        """
        centroid: "np.ndarray" = np.empty((index.size, X.shape[1]))
        for i in range(len(index)):
            centroid[i] = np.mean(X[inverse==i])
            
        return centroid


class Annoy():
    
    @staticmethod
    def build_annoy(data: "np.ndarray",
                    metric: str = "angular",
                    n_trees: int=10) -> "AnnoyIndex":
        """Build ``AnnoyIndex`` object from data.

        :param data: The data array
        :param metric: The distance metric to use, defaults to "angular"
        :param n_trees: The number of trees, defaults to 10
        
        :return: An ``AnnoyIndex`` object.
        """
        
        model: "AnnoyIndex" = AnnoyIndex(data.shape[1], metric = metric) #type: ignore
        for i in range(data.shape[0]):
            model.add_item(i, data[i])
        model.build(n_trees=n_trees, n_jobs=-1)
        return model
    
    
    @staticmethod
    def load_annoy(path: str,
                   ncol: int,
                   metric: str = "angular"
                   ) -> "AnnoyIndex":
        """Load ``AnnoyIndex`` object from disk.
        
        This loads an AnnoyIndex object saved using this class or the ANnoy's buildin IO function.   

        :param path: The path to the object.
        :param ncol: The number of columns.
        :param metric: _description_, defaults to "angular"
        :return: The loaded ``AnnoyIndex`` object.
        """
        model: "AnnoyIndex" = AnnoyIndex(ncol, metric) #type: ignore
        model.load(path)
        return model
    
    
    @staticmethod
    def save_annoy(model: "AnnoyIndex", path: str):
        """Save ``AnnoyIndex`` object to disk.
        
        This saves an ``AnnoyIndex`` object to a specified path.   

        :param model: An ``AnnoyIndex`` object to be saved.
        :param path: The path to the object.
        :return: The loaded ``AnnoyIndex`` object.
        """
        model.save(path)