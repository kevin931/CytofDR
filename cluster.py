import phenograph as pg
import numpy as np
from scipy.sparse.base import spmatrix

from typing import Tuple, Dict, List, Union

class Cluster():
    
    @classmethod
    def cluster(cls, data: "np.ndarray", methods: Union[str, List[str]]):
        
        if not isinstance(methods, list):
            methods = [methods]
        
        labels: Dict[str, "np.ndarray"] = dict()
        
        if "phenograph" in methods:
            labels["phenograph"] = cls.phenograph(data)[0]
                
        return labels
    
    
    @classmethod
    def save_results(cls, labels: Dict[str, "np.ndarray"], out: str) -> None:
        
        method: str
        label: "np.ndarray"
        
        for method, label in labels.items():
            save_path = "{}/{}.txt".format(out, method)
            np.savetxt(save_path, label, delimiter="\t")
        
    
    @staticmethod
    def phenograph(data: "np.ndarray") -> Tuple["np.ndarray", "spmatrix", float]:
        
        communities: "np.ndarray"
        graph: "spmatrix"
        Q: float        
        communities, graph, Q = pg.cluster(data=data)
            
        return communities, graph, Q