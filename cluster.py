import phenograph as pg
import numpy as np
from scipy.sparse.base import spmatrix

from fileio import FileIO

from typing import Tuple, Dict, List, Union, Optional

class Cluster():
    
    @classmethod
    def cluster(cls,
                data: "np.ndarray",
                methods: Union[str, List[str]],
                out: Optional[str]=None):
        
        if not isinstance(methods, list):
            methods = [methods]
        
        labels: Dict[str, "np.ndarray"] = dict()
        
        if "phenograph" in methods:
            labels["phenograph"] = cls.phenograph(data)[0]
            
        if out is not None:
            cls.save_results(labels=labels, out=out)
                
        return labels
    
    
    @classmethod
    def save_results(cls,
                     labels: Dict[str, "np.ndarray"],
                     out: str) -> None:
        
        method: str
        label: "np.ndarray"
        
        for method, label in labels.items():
            FileIO.save_np_array(array=label, dir_path=out, file_name=method)
        
    
    @staticmethod
    def phenograph(data: "np.ndarray") -> Tuple["np.ndarray", "spmatrix", float]:
        
        communities: "np.ndarray"
        graph: "spmatrix"
        Q: float        
        communities, graph, Q = pg.cluster(data=data)
            
        return communities, graph, Q