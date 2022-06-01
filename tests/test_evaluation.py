from CytofDR import evaluation
import numpy as np
import annoy
import scipy.spatial
import sklearn.neighbors

import pytest
import os
import shutil

from typing import Optional, Union


class TestEvaluationMetrics():
    
    @classmethod
    def setup_class(cls):
        cls.data: np.ndarray = np.abs(np.random.rand(100, 10))
        cls.data_labels: np.ndarray = np.repeat(["a", "b"], 50)
        cls.embedding: np.ndarray = np.random.rand(100, 2)
        cls.embedding_labels: np.ndarray = np.repeat(["a", "b", "c", "d"], 25)
        cls.a: np.ndarray = np.array([1, 2, 3])
        cls.b: np.ndarray = np.array([4, 5, 6])
        cls.neighbors_data: np.ndarray = sklearn.neighbors.NearestNeighbors().fit(cls.data).kneighbors(cls.data, return_distance=False)
        cls.neighbors_embedding: np.ndarray = sklearn.neighbors.NearestNeighbors().fit(cls.embedding).kneighbors(cls.embedding, return_distance=False)
        cls.data_pwd = scipy.spatial.distance.pdist(cls.data)
        
        os.mkdir("./tmp_pytest/")
        
        
    @pytest.mark.parametrize("k", [5, 10])
    def test_build_annoy(self, k: int):
        neighbors: np.ndarray = evaluation.EvaluationMetrics.build_annoy(self.data, k=k)
        assert isinstance(neighbors, np.ndarray)
        assert neighbors.shape == (100, k)
        
        
    def test_build_annoy_path(self):
        annoy_object: annoy.AnnoyIndex = evaluation.Annoy.build_annoy(self.data)
        evaluation.Annoy.save_annoy(annoy_object, path = "./tmp_pytest/annoy_test.txt")
        neighbors: np.ndarray = evaluation.EvaluationMetrics.build_annoy(self.data, saved_annoy_path="./tmp_pytest/annoy_test.txt")
        assert isinstance(neighbors, np.ndarray)
        assert neighbors.shape == (100, 5)
        
        
    @pytest.mark.parametrize("metric", ["Pearson", "Spearman"])
    def test_correlation(self, metric: str):
        assert np.isclose(evaluation.EvaluationMetrics.correlation(self.a, self.b, metric), 1)
        
        
    def test_residual_variance(self):
        assert np.isclose(evaluation.EvaluationMetrics.residual_variance(self.a, self.b), 0)
        
    
    @pytest.mark.parametrize("r", [1.1, -1.1])
    def test_residual_variance_r_value_error(self, r: float):        
        try:
            evaluation.EvaluationMetrics.residual_variance(r=r)
        except ValueError as e:
            assert "'r' must be between -1 and 1." in str(e)
        
    
    @pytest.mark.parametrize("x,y", [(np.array([1, 2, 3]), None),
                                     (None, np.array([1, 2, 3])),
                                     (None, None)])
    def test_residual_variance_missing_xy(self, x: Optional[np.ndarray], y: Optional[np.ndarray]):
        try:
            evaluation.EvaluationMetrics.residual_variance(x=x, y=y)
        except ValueError as e:
            assert "Either 'r' or both 'x' and 'y' is needed." in str(e)
        
        
    def test_emd(self):
        emd: float = evaluation.EvaluationMetrics.EMD(self.a, self.b)
        assert isinstance(emd, float)
        assert emd > 0
        

    def test_KNN(self):
        knn: float = evaluation.EvaluationMetrics.KNN(self.neighbors_data, self.neighbors_embedding)
        assert isinstance(knn, float)
        assert knn >= 0 and knn <=1
    
    def test_NPE(self):
        npe: float = evaluation.EvaluationMetrics.NPE(self.neighbors_data, self.neighbors_embedding, self.data_labels)
        assert isinstance(npe, float)
        assert npe >= 0
    
    
    def test_neighborhood_agreement(self):
        agreement: float = evaluation.EvaluationMetrics.neighborhood_agreement(self.neighbors_data, self.neighbors_embedding)
        assert isinstance(agreement, float)
        
    
    def test_neighborhood_trustworthiness(self):
        trust: float = evaluation.EvaluationMetrics.neighborhood_trustworthiness(self.neighbors_data, self.neighbors_embedding, self.data_pwd)
        assert isinstance(trust, float)
        assert trust >= 0
        
        
    def test_random_forest(self):
        accuracy: float = evaluation.EvaluationMetrics.random_forest(self.embedding, self.data_labels)
        assert isinstance(accuracy, float)
        assert accuracy >= 0 and accuracy <=1
        
        
    def test_silhouette(self):
        sil: float = evaluation.EvaluationMetrics.silhouette(self.embedding, self.data_labels)
        assert isinstance(sil, float)
        assert sil >= -1 and sil <=1
        
        
    def test_davies_bouldin(self):
        dbi: float = evaluation.EvaluationMetrics.davies_bouldin(self.embedding, self.data_labels)
        assert isinstance(dbi, float)
        assert dbi >= 0
        
        
    def test_calinski_harabasz(self):
        chi: float = evaluation.EvaluationMetrics.calinski_harabasz(self.embedding, self.data_labels)
        assert isinstance(chi, float)
        assert chi >= 0
        
        
    def test_ARI(self):
        ari: float = evaluation.EvaluationMetrics.ARI(self.embedding_labels, self.data_labels)
        assert isinstance(ari, float)
        assert ari >= 0 and ari <= 1
        
        
    def test_NMI(self):
        nmi: float = evaluation.EvaluationMetrics.NMI(self.embedding_labels, self.data_labels)
        assert isinstance(nmi, float)
        assert nmi >= 0 and nmi <=2
        
        
    @pytest.mark.parametrize("method", ["emd", "cluster_distance"])
    def test_embedding_concordance(self, method: str):
        score: Union[float, str] = evaluation.EvaluationMetrics.embedding_concordance(self.embedding, self.embedding_labels,
                                                                                      self.data, self.data_labels, ["a", "b"],
                                                                                      method=method)
        assert isinstance(score, float)
        assert score >= 0
        
        
    def test_embedding_concordance_comparison_labels(self):
        score: Union[float, str] = evaluation.EvaluationMetrics.embedding_concordance(self.embedding, self.embedding_labels,
                                                                                      self.data, self.data_labels)
        assert isinstance(score, float)
        assert score >= 0
        
        
    def test_embedding_concordance_na(self):
        score: Union[float, str] = evaluation.EvaluationMetrics.embedding_concordance(self.embedding, self.embedding_labels,
                                                                                      self.data, self.data_labels, ["e"])
        assert isinstance(score, str)
        assert score == "NA"
    
    
    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp_pytest/")


class TestPointClusterDistance():
    
    @classmethod
    def setup_class(cls):
        cls.data: np.ndarray = np.abs(np.random.rand(100, 10))
        cls.labels: np.ndarray = np.repeat(["a", "b", "c", "d"], 25)
    
    
    def test_flatten(self):
        flattened_data: np.ndarray = evaluation.PointClusterDistance.flatten(self.data)
        assert flattened_data.shape[0] == 1000
    
    
    def test_pcd_fit(self):
        dist: np.ndarray= evaluation.PointClusterDistance(self.data, self.labels).fit(flatten=False)
        assert dist.shape == (100, 4)
    
    
    def test_pcd_fit_flatten(self):
        dist: np.ndarray= evaluation.PointClusterDistance(self.data, self.labels).fit(flatten=True)
        assert dist.shape[0] == 400
        
    
    @pytest.mark.parametrize("metric", ["manhattan", "cosine"])
    def test_pcd_dist(self, metric: str):
        dist: np.ndarray= evaluation.PointClusterDistance(self.data, self.labels, dist_metric=metric).fit(flatten=False)
        assert dist.shape == (100, 4)
        


class TestAnnoy():
    
    @classmethod
    def setup_class(cls):
        cls.data: np.ndarray = np.abs(np.random.rand(100, 10))
        os.mkdir("./tmp_pytest/")
        
    
    @pytest.mark.parametrize("annoy_metric,n_trees", [("angular", 5),
                                                      ("angular", 10),
                                                      ("euclidean", 5),
                                                      ("euclidean", 10)])
    def test_build_annoy(self, annoy_metric: str, n_trees: int):
        annoy_object: annoy.AnnoyIndex = evaluation.Annoy.build_annoy(self.data, metric = annoy_metric, n_trees = n_trees)
        assert isinstance(annoy_object, annoy.AnnoyIndex)
        
        
    def test_save_annoy(self):
        annoy_object: annoy.AnnoyIndex = evaluation.Annoy.build_annoy(self.data)
        evaluation.Annoy.save_annoy(annoy_object, path = "./tmp_pytest/annoy_test.txt")
        assert True
        
    
    def test_load_annoy(self):
        annoy_object: annoy.AnnoyIndex = evaluation.Annoy.load_annoy("./tmp_pytest/annoy_test.txt", ncol = 10)
        assert isinstance(annoy_object, annoy.AnnoyIndex)
        
        
    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp_pytest/")