import numpy as np
from annoy import AnnoyIndex

from fileio import FileIO
from typing import List, Optional, Tuple

class Annoy():
    
    @staticmethod
    def build_annoy(data: "np.ndarray",
                    metric: str = "angular",
                    n_trees: int=10) -> "AnnoyIndex":
        
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
        
        model: "AnnoyIndex" = AnnoyIndex(ncol, metric) #type: ignore
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
    
    
    @staticmethod
    def train_test_split(data: "np.ndarray",
                         train_percent: float,
                         col_names: Optional["np.ndarray"]=None,
                         out: Optional[str]=None) -> Tuple["np.ndarray", "np.ndarray"]:
        
        data_train: "np.ndarray"
        data_test: "np.ndarray"
        data_train_size: int = int(data.shape[0]*train_percent)
        data_index: "np.ndarray" = np.arange(data.shape[0])
        
        data_train_index: "np.ndarray" = np.random.choice(data_index, data_train_size, replace=False)
        data_test_index: "np.ndarray" = np.delete(data_index, data_train_index)
        
        data_train = data[data_train_index]
        data_test = data[data_test_index]
        
        if out is not None:
            FileIO.save_np_array(data_train, out, "train", col_names=col_names)
            FileIO.save_np_array(data_test, out, "test", col_names=col_names)
            FileIO.save_np_array(data_train_index, out, "train_index")
            FileIO.save_np_array(data_test_index, out, "test_index")
            
        return data_train, data_test
            
        