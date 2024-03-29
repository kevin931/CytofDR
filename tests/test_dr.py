from CytofDR import dr
import numpy as np

import sys
from io import StringIO
import os
import shutil
import pytest
from typing import Dict, Union, List, Optional

METHODS: Dict[str, bool] = {"SAUCIE": True, "ZIFA": True, "GrandPrix": True}

try:
    import SAUCIE
except ImportError:
    METHODS["SAUCIE"] = False
    
try:
    from ZIFA import ZIFA
except ImportError:
    METHODS["ZIFA"] = False
     
try:
    from GrandPrix import GrandPrix
except ImportError:
    METHODS["GrandPrix"] = False


class TestDRMethods():
    
    @classmethod
    def setup_class(cls):
        cls.expression: np.ndarray = np.abs(np.random.normal(2, 1, (35, 10)))
        cls.expression[[0, 0]] = 0
        cls.transform: np.ndarray = np.abs(np.random.normal(2, 1, (10, 10)))
        cls.expression[[0, 0]] = 0
        
        
    @pytest.mark.parametrize("method,out_dim", [("PCA", 2),
                                                ("ICA", 2),
                                                ("FA", 2),
                                                ("NMF", 2),
                                                ("ZIFA", 2)])
    def test_linear_methods(self, method: str, out_dim: int):
        embedding: np.ndarray = getattr(dr.LinearMethods, method)(self.expression, out_dim)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (35, out_dim)
    
    
    @pytest.mark.parametrize("method,out_dim", [("MDS", 2),
                                                ("UMAP", 2),
                                                ("isomap", 2),
                                                ("LLE", 2),
                                                ("spectral", 2),
                                                ("open_tsne", 2),
                                                ("sklearn_tsne", 2),
                                                ("phate", 2),
                                                ("spectral", 2)])
    def test_non_linear_methods(self, method: str, out_dim: int):
        embedding: np.ndarray = getattr(dr.NonLinearMethods, method)(self.expression, out_dim)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (35, out_dim)
        
    
    @pytest.mark.parametrize("kernel,out_dim", [("poly", 2),
                                                ("linear", 2),
                                                ("rbf", 2),
                                                ("cosine", 2),
                                                ("sigmoid", 2)])
    def test_kernelPCA(self, kernel: str, out_dim: int):
        embedding: np.ndarray = getattr(dr.NonLinearMethods, "kernelPCA")(self.expression, out_dim, kernel)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (35, out_dim)
        
        
    @pytest.mark.parametrize("method", ["isomap", "LLE"])
    def test_transform(self, method: str):
        embedding: np.ndarray = getattr(dr.NonLinearMethods, method)(self.expression, transform=self.transform)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (10, 2)
        
        
    def test_open_tsne_perp_list(self):
        embedding: np.ndarray = dr.NonLinearMethods.open_tsne(self.expression, perp=[5, 10])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (35, 2)
        
    
    @pytest.mark.parametrize("init", ["spectral", "random", np.random.normal(size=(35, 2))])
    def test_open_tsne_init(self, init: Union[str, np.ndarray]): 
        embedding: np.ndarray = dr.NonLinearMethods.open_tsne(self.expression, init=init)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (35, 2)
    
    
    @pytest.mark.parametrize("out_dim", [2, 3])
    def test_GrandPrix(self, out_dim: int):
        if METHODS["GrandPrix"]:
            embedding: np.ndarray = dr.NonLinearMethods.grandprix(self.expression, out_dims=out_dim)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (35, out_dim)
            
    
    def test_SAUCIE(self):
        if METHODS["SAUCIE"]:
            embedding: np.ndarray = dr.NonLinearMethods.SAUCIE(self.expression)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (35, 2)
            
            
    def test_run_dr_methods_all(self):
        out: dr.Reductions = dr.run_dr_methods(data=self.expression, methods="all")
        assert isinstance(out, dr.Reductions)
        
        methods = ["PCA", "ICA", "UMAP", "sklearn_tsne", "open_tsne", "FA", "Isomap", "MDS", "LLE",
                   "kpca_poly", "kpca_rbf", "PHATE", "NMF", "Spectral"]
        if METHODS["ZIFA"]:
            methods.append("ZIFA")
        if METHODS["GrandPrix"]:
            methods.append("GrandPrix")
            
        out_methods = list(out.reductions.keys())
        for m in methods:
            assert m in out_methods
            
            
    @pytest.mark.parametrize("method", ["PCA", "ICA", "ZIFA", "NMF", "FA",
                                        "MDS", "UMAP", "open_tsne", "sklearn_tsne",
                                        "PHATE", "Spectral", "Isomap", "LLE"])  
    def test_run_dr_method(self, method):
        out: dr.Reductions = dr.run_dr_methods(data=self.expression, methods=method)
        assert isinstance(out, dr.Reductions)
        assert method in list(out.reductions.keys())
    
   
    def test_run_dr_method_grandprix(self):
        if METHODS["GrandPrix"]:
            out: dr.Reductions = dr.run_dr_methods(data=self.expression, methods="GrandPrix")
            assert isinstance(out, dr.Reductions)
            assert "GrandPrix" in list(out.reductions.keys())
            
            
    @pytest.mark.parametrize("method", ["Isomap", "LLE"])
    def test_run_dr_method_transform(self, method):
        out: dr.Reductions = dr.run_dr_methods(data=self.expression, methods=method,transform=self.transform)
        assert isinstance(out, dr.Reductions)
        assert method in list(out.reductions.keys())
        assert out.reductions[method].shape == (10, 2)
        
        
class TestReductions():
    
    @classmethod
    def setup_class(cls):
        cls.embedding: np.ndarray = np.random.rand(10, 2)
        cls.results = dr.Reductions({"test_dr": cls.embedding})
        os.mkdir("./tmp_pytest/")
        
    
    def test_names_attributes(self):
        assert isinstance(self.results.names, set)
        assert "test_dr" in self.results.names
        
        
    def test_get_item(self):
        reduction = self.results["test_dr"]
        assert isinstance(reduction, np.ndarray)
        assert reduction.shape == (10, 2)
        
    
    def test_get_reduction(self):
        reduction = self.results.get_reduction("test_dr")
        assert isinstance(reduction, np.ndarray)
        assert reduction.shape == (10, 2)
        
        
    def test_reduction_setter(self):
        reduction  = self.results.get_reduction("test_dr")
        self.results.reductions = {"test_dr": reduction}
        
        
    def test_add_item(self):
        self.results["new_dr"] = np.random.rand(10, 2)
        assert "new_dr" in self.results.names
        assert self.results.get_reduction("new_dr").shape == (10, 2)
        
        
    def test_del_item(self):
        del self.results["new_dr"]
        assert len(self.results.reductions) == 1
        assert "new_dr" not in self.results.names
        
        
    def test_add_reduction(self):
        new_embedding: np.ndarray = np.random.rand(10, 2)
        self.results.add_reduction(new_embedding, "new_dr")
        assert "new_dr" in list(self.results.reductions.keys())
        assert self.results.get_reduction("new_dr").shape == (10, 2)
        
        
    def test_add_reduction_no_replace_error(self):
        try:
            self.results.add_reduction(self.embedding, "test_dr")
        except ValueError as e:
            assert "Reduction already exists. Set 'replace' to True if replacement is intended." in str(e)
        else:
            assert False
            
        
    def test_add_metadata(self):
        
        original_data = np.abs(np.random.rand(10, 10))
        original_labels = np.repeat(np.array([1,2]), 5)
        original_cell_types = np.repeat(np.array(["a","b"]), 5)
        embedding_labels = {"test_dr": np.repeat(np.array([1,2,3,4,5]), 2), "new_dr": np.repeat(np.array([1,2,3,4,5]), 2)}
        embedding_cell_types = {"test_dr": np.repeat(np.array(["a","b", "c", "d", "e"]), 2), "new_dr": np.repeat(np.array(["a","b", "c", "d", "e"]), 2)}
        comparison_data = np.abs(np.random.rand(10, 12))
        comparison_cell_types = np.repeat(np.array(["a","b"]), 5)
        comparison_classes = ["a", "b"]
        
        self.results.add_evaluation_metadata(original_data=original_data,
                                             original_labels=original_labels,
                                             original_cell_types=original_cell_types,
                                             embedding_labels=embedding_labels,
                                             embedding_cell_types=embedding_cell_types,
                                             comparison_data=comparison_data,
                                             comparison_cell_types=comparison_cell_types,
                                             comparison_classes=comparison_classes)
        
        assert isinstance(self.results.original_data, np.ndarray)
        assert isinstance(self.results.original_labels, np.ndarray)
        assert isinstance(self.results.original_cell_types, np.ndarray)
        assert isinstance(self.results.embedding_labels, dict)
        assert isinstance(self.results.embedding_cell_types, dict)
        assert isinstance(self.results.comparison_data, np.ndarray)
        assert isinstance(self.results.comparison_classes, list)
        assert isinstance(self.results.comparison_cell_types, np.ndarray)
        
        
    def test_add_metadata_none(self):
        
        self.results.add_evaluation_metadata()
        
        assert isinstance(self.results.original_data, np.ndarray)
        assert isinstance(self.results.original_labels, np.ndarray)
        assert isinstance(self.results.original_cell_types, np.ndarray)
        assert isinstance(self.results.embedding_labels, dict)
        assert isinstance(self.results.embedding_cell_types, dict)
        assert isinstance(self.results.comparison_data, np.ndarray)
        assert isinstance(self.results.comparison_classes, list)
        assert isinstance(self.results.comparison_cell_types, np.ndarray)
        
        
    def test_evaluate_category(self):
        screen_stdout = sys.stdout
        string_stdout = StringIO()
        sys.stdout = string_stdout
        
        self.results.evaluate(category=["global", "local", "downstream", "concordance"])
        categories: List[str] = ["global","local", "downstream", "concordance"]
        eval_categories: List[str] = list(self.results.evaluations.keys())
        
        output = string_stdout.getvalue()
        sys.stdout = screen_stdout
        print(output)
        
        for c in categories:
            assert c in eval_categories
            assert "Evaluating " + c in output
                  
        
    def test_rank_dr_methods(self):
        results: Dict[str, float] = self.results.rank_dr_methods()
        for m in ["new_dr", "test_dr"]: 
            assert m in results.keys()
            assert isinstance(results[m], float)
            assert results[m] <= 2 and results[m] >= 1
            
            
    def test_rank_dr_methods_no_type_cluster(self):
        self.results.original_cell_types = None
        self.results.evaluate(category=["downstream"])
        results: Dict[str, float] = self.results.rank_dr_methods()
        for m in ["new_dr", "test_dr"]: 
            assert m in results.keys()
            assert isinstance(results[m], float)
            assert results[m] <= 2 and results[m] >= 1
        
        self.results.original_cell_types = np.repeat(np.array(["a","b"]), 5)
            
            
    def test_evaluate_pairwise(self):
        self.results.evaluate(category="global", pwd_metric="pairwise")
        assert "global" in self.results.evaluations.keys()
        
        
    def test_evaluate_npe_method(self):
        self.results.evaluate(category="global", NPE_method="tvd")
        assert "global" in self.results.evaluations.keys()
        
        
    def test_evaluate_normalize_pwd(self):
        self.results.evaluate(category="global", normalize_pwd="minmax")
        assert "global" in self.results.evaluations.keys()
        
        
    def test_evaluate_pairwise_downsample(self):
        self.results.evaluate(category="global", pwd_metric="pairwise_downsample", pairwise_downsample_size=5)
        assert "global" in self.results.evaluations.keys()
        
        
    def test_evaluate_pairwise_downsample_default(self):
        self.results.evaluate(category="global", pwd_metric="pairwise_downsample")
        assert "global" in self.results.evaluations.keys()
        
        
    def test_evaluate_k_neighbors(self):
        self.results.evaluate(category="local", k_neighbors=3)
        assert "local" in self.results.evaluations.keys()
        
        
    def test_evaluate_metric_error(self):
        try:
            self.results.evaluate(category="global", pwd_metric="Wrong")
        except ValueError as e:
            assert "Unsupported 'pwd_metric': 'PCD', 'Pairwise', or 'pairwise_downsample' only." in str(e)
        else:
            assert False
            
            
    def test_evaluate_no_reduction_error(self):
        reduction: dr.Reductions = dr.Reductions()
        try:
            reduction.evaluate(category="global")
        except ValueError as e:
            assert "No reductions to evalate. Add your reductions first." in str(e)
        else:
            assert False
            
            
    def test_cluster(self):
        self.results.original_labels = None
        self.results.embedding_labels = None
        assert self.results.original_labels is None
        assert self.results.embedding_labels is None
        
        self.results.cluster(n_clusters=3)
        assert isinstance(self.results.original_labels, np.ndarray)
        assert isinstance(self.results.embedding_labels, dict)
        assert np.unique(self.results.original_labels).shape[0] == 3
        assert np.unique(self.results.embedding_labels["test_dr"]).shape[0] == 3
        
        
    def test_cluster_data_error(self):
        results = dr.Reductions({"new_dr": self.embedding})
        results.original_labels = np.repeat(np.array([1,2]), 5)
        try:
            results.cluster(n_clusters=3)
        except ValueError as e:
            assert "'original_data' is missing: cannot cluster." in str(e)
        else:
            assert False
            

    def test_cluster_embedding_error(self):
        results = dr.Reductions()
        try:
            results.cluster(n_clusters=3, cluster_data=False)
        except ValueError as e:
            assert "'reductions' is missing: cannot cluster." in str(e)
        else:
            assert False
            
                        
    @pytest.mark.parametrize("original_labels, embedding_labels",
                             [(None, {"new_dr": np.repeat(np.array([1,2]), 5)}),
                              (np.repeat(np.array([1,2]), 5), None)])
    def test_evaluate_no_labels(self, original_labels: Optional[np.ndarray], embedding_labels: Optional[dict]):
        results = dr.Reductions({"new_dr": self.embedding})
        results.original_data = np.random.rand(10, 2)
        results.original_labels = original_labels
        results.embedding_labels = embedding_labels
        try:
            results.evaluate(category="global", auto_cluster=False)
        except ValueError as e:
            assert "Evaluation needs 'original_labels', and 'embedding_labels' attributes. " in str(e)
        else:
            assert False
            
         
    def test_evaluate_no_original_data(self):
        results = dr.Reductions({"new_dr": self.embedding})
        results.original_data = None
        results.original_labels = np.repeat(np.array([1,2]), 5)
        results.embedding_labels = {"new_dr": np.repeat(np.array([1,2]), 5)}
        try:
            results.evaluate(category="global", auto_cluster=True)
        except ValueError as e:
            assert "Evaluation needs 'original_data'. Please add it with the 'add_evaluation_metadata()' method." in str(e)
        else:
            assert False   
            

    def test_evaluate_auto_cluster(self):
        results = dr.Reductions({"new_dr": self.embedding})
        results.original_data = np.random.rand(10, 2)
        results.evaluate(category=["global", "local", "downstream"], n_clusters=2)
        assert self.results.evaluations is not None
        
    
    def test_custom_evaluate(self):
        self.results.add_custom_evaluation_result("m1", "test_dr", 0.5)
        self.results.add_custom_evaluation_result("m1", "new_dr", 0.2)
        self.results.add_custom_evaluation_result("m2", "test_dr", 0.3)
        self.results.add_custom_evaluation_result("m2", "new_dr", 0.1)
        self.results.add_custom_evaluation_result("m3", "test_dr", 0.3, reverse_ranking=True)
        self.results.add_custom_evaluation_result("m3", "new_dr", 0.1, reverse_ranking=True)
        assert len(self.results.custom_evaluations["custom"]) == 3
        assert np.isclose(self.results.custom_evaluations["custom"]["m1"]["new_dr"], 0.2)
        
        
    def test_rank_dr_custom(self):
        ranking: Dict[str, float] = self.results.rank_dr_methods_custom()
        assert isinstance(ranking, dict)
        assert np.isclose(ranking["test_dr"], 5/3)
        assert np.isclose(ranking["new_dr"], 4/3)
        
    
    def test_custom_evaluate_weights(self):
        self.results.custom_evaluations = {"custom": {}}
        self.results.custom_evaluation_reverse_ranking = {}
        self.results.add_custom_evaluation_result("m1", "test_dr", 0.5, 0.8)
        self.results.add_custom_evaluation_result("m1", "new_dr", 0.2, 0.8)
        self.results.add_custom_evaluation_result("m2", "test_dr", 0.3, 0.2, reverse_ranking=True)
        self.results.add_custom_evaluation_result("m2", "new_dr", 0.1, 0.2, reverse_ranking=True)
        assert np.isclose(self.results.custom_evaluation_weights["m1"], 0.8)
        assert np.isclose(self.results.custom_evaluation_weights["m2"], 0.2)
        
        
    def test_rank_dr_custom_weights(self):
        ranking: Dict[str, float] = self.results.rank_dr_methods_custom()
        assert np.isclose(ranking["test_dr"], 1.8)
        assert np.isclose(ranking["new_dr"], 1.2)
        
        
    def test_custom_rank_missing_method(self):
        self.results.custom_evaluations = {"custom": {}}
        self.results.custom_evaluation_reverse_ranking = {}
        self.results.add_custom_evaluation_result("m1", "test_dr", 0.5, 0.8)
        self.results.add_custom_evaluation_result("m1", "new_dr", 0.2, 0.8)
        self.results.add_custom_evaluation_result("m2", "test_dr", 0.3, 0.2)
        
        try:
            self.results.rank_dr_methods_custom()
        except RuntimeError as e:
            assert "Missing metrics for the following methods:" in str(e)
        else:
            assert False


    def test_plot_reduction(self):
        self.results.plot_reduction("test_dr", "./tmp_pytest/test_plot.png")
        assert os.path.exists("./tmp_pytest/test_plot.png")
            
            
    def test_save_all_reductions(self):
        self.results.save_all_reductions("./tmp_pytest/")
        assert os.path.exists("./tmp_pytest/test_dr.txt")
        assert os.path.exists("./tmp_pytest/new_dr.txt")
        
        
    def test_save_reduction_file_exists_error(self):
        try:
            self.results.save_reduction("new_dr", "./tmp_pytest/new_dr.txt")
        except FileExistsError:
            assert True
        else:
            assert False
            
            
    def test_save_reduction_new_name(self):
        self.results.save_reduction("new_dr", "./tmp_pytest/new_dr2.txt")
        assert os.path.exists("./tmp_pytest/new_dr2.txt")
        
        
    def test_save_evaluations(self):
        self.results.save_evaluations("./tmp_pytest/evaluations.txt")
        assert os.path.exists("./tmp_pytest/new_dr2.txt")
        
        
    def test_save_evaluations_file_exists(self):
        try:
            self.results.save_evaluations("./tmp_pytest/evaluations.txt")
        except FileExistsError:
            assert True
        else:
            assert False
            
            
    def test_save_evaluations_no_evaluations(self):
        self.results.evaluations = {}
        try:
            self.results.save_evaluations("./tmp_pytest/evaluations2.txt")
        except AttributeError as e:
            assert "No 'evaluations' values found. Run the 'evaluate()' method first." in str(e)
        else:
            assert False
            
            
    def test_str(self):
        out_str: str = str(self.results)
        assert out_str == f"A 'Reductions' object with {', '.join(self.results.names)}."
        
        
    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp_pytest/")


def test_verbose():
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    dr._verbose("Test verbose", verbose=True)
    
    output = string_stdout.getvalue()
    sys.stdout = screen_stdout
    assert "Test verbose" in output
    

def test_verbose_off():
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    dr._verbose("Test verbose", verbose=False)
    
    output = string_stdout.getvalue()
    sys.stdout = screen_stdout
    assert output == ""
    