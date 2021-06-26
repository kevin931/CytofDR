import numpy as np
from annoy import AnnoyIndex

from fileio import FileIO

from typing import Literal, List

class Annoy():
    
    @staticmethod
    def build_annoy(data: "np.ndarray",
                    metric: Literal["angular", "euclidean", "manhattan", "dot", "hamming"] = "angular",
                    n_trees: int=10) -> "AnnoyIndex":
        
        model: "AnnoyIndex" = AnnoyIndex(data.shape[1], metric = metric)
        for i in range(data.shape[0]):
            model.add_item(i, data[i])
        model.build(n_trees=n_trees, n_jobs=-1)
        return model
    
    
    @staticmethod
    def load_annoy(path: str,
                   ncol: int,
                   metric: Literal["angular", "euclidean", "manhattan", "dot", "hamming"] = "angular"
                   ) -> "AnnoyIndex":
        
        model: "AnnoyIndex" = AnnoyIndex(ncol, metric)
        model.load(path)
        return model
    
    
    @staticmethod
    def save_annoy(model: "AnnoyIndex", path: str):
        model.save(path)
        
        
class DownSample():
    
    @staticmethod
    def downsample_from_data(data: "np.ndarray",
                             n: int,
                             n_fold: int=1,
                             save_downsample_index: str=None) -> List["np.ndarray"]:
        
        index_list = []
        i: int
        for i in range(n_fold):
            index: "np.ndarray" = np.random.choice(data.shape[0], size=n)
            index_list.append(index)
            if save_downsample_index is not None:
                index_name: str = "index_{}".format(i)
                FileIO.save_np_array(index, save_downsample_index, file_name=index_name)
                
        return index_list