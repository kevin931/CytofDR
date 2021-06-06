import numpy as np
from numpy.lib.function_base import _corrcoef_dispatcher
import dr
import metric
import cluster
from fileio import FileIO

import argparse
from typing import Optional, Union, List, Dict, Any


def main(cmdargs: Dict[str, Any]):
    
    data: List["np.ndarray"] = FileIO.load_data(files=cmdargs["file"],
                                                col_names=cmdargs["file_col_names"],
                                                add_sample_index=cmdargs["add_sample_index"],
                                                drop_columns=cmdargs["file_drop_col"],
                                                delim=cmdargs["delim"])
    
    if cmdargs["evaluate"]:
        
        embedding: List["np.ndarray"] = FileIO.load_data(files=cmdargs["embedding"],
                                                         col_names=cmdargs["embedding_col_names"],
                                                         concat=cmdargs["concat"],
                                                         add_sample_index=cmdargs["add_sample_index"],
                                                         drop_columns=cmdargs["embedding_drop_col"])
                 
        label: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        
        if cmdargs["label"] is not None:
            
            label = FileIO.load_data(files=cmdargs["label"],
                                     col_names=cmdargs["label_col_names"],
                                     concat=cmdargs["concat"],
                                     add_sample_index=cmdargs["add_sample_index"],
                                     drop_columns=cmdargs["label_drop_col"])[1]

        results: List[List[Union[str, float]]]
        
        embedding_names: "np.ndarray"
        
        if len(cmdargs["embedding"]) == 1:
            embedding_names = np.array(FileIO._check_dir(cmdargs["embedding"][0]))
        else:
            embedding_names = np.array(cmdargs["embedding"])
                      
        if cmdargs["downsample"] is None:
        
            results= metric.Metric.run_metrics(data=data[1],
                                               embedding=embedding,
                                               methods=cmdargs["methods"],
                                               labels=label,
                                               embedding_names=embedding_names)
            
        else:
            
            results = metric.Metric.run_metrics_downsample(data=data[1],
                                                           embedding=embedding,
                                                           methods=cmdargs["methods"],
                                                           labels=label,
                                                           embedding_names=embedding_names,
                                                           downsample=cmdargs["downsample"],
                                                           n_fold=cmdargs["k_fold"])

        FileIO.save_list_to_csv(results, cmdargs["out"], "eval_metric")
        
    
    if cmdargs["cluster"]:
        cluster.Cluster.cluster(data[1],
                                methods=cmdargs["methods"],
                                out=cmdargs["out"])
    
    
    if cmdargs["dr"]:
        dr.DR.run_methods(data=data[1],
                          out=cmdargs["out"],
                          methods=cmdargs["methods"],
                          out_dims=cmdargs["out_dims"],
                          perp=cmdargs["perp"],
                          early_exaggeration=cmdargs["early_exaggeration"],
                          early_exaggeration_iter=cmdargs["early_exaggeration_iter"],
                          tsne_learning_rate=cmdargs["tsne_learning_rate"],
                          max_iter=cmdargs["max_iter"],
                          init=cmdargs["init"],
                          open_tsne_method=cmdargs["open_tsne_method"])
        
        
class _Arguments():
    # This class parses command line arguments.

    def __init__(self) -> None:
        
        self.parser = argparse.ArgumentParser(description="Command Line Arguments")
        
        # Program Mode
        self.parser.add_argument("--cluster", action="store_true",
                                 help="Cluster the input file.")
        self.parser.add_argument("--evaluate", action="store_true",
                                 help="Evaluate embedding results.")
        self.parser.add_argument("--dr", action="store_true",
                                 help="Running dimension reduction algorithms.")          

        # Methods: For all modes
        self.parser.add_argument("-m", "--methods", nargs="+", action="store",
                                 help="Methods to run: applies to all modules.")
        
        # File IO
        self.parser.add_argument("-f", "--file", nargs="+", action="store",
                                 help="Path to directory or original files.")
        self.parser.add_argument("--concat", action="store_true",
                                 help="Concatenate files, embeddings, and labels read.")
        self.parser.add_argument("--delim", type=str, action="store",
                                 help="File delimiter.")
        self.parser.add_argument("-o", "--out", type=str, action="store",
                                 help="Directory name for saving results")
        self.parser.add_argument("--no_new_dir", action="store_true",
                                 help="Save results directly in -o without creating new directory.")
        self.parser.add_argument("--embedding", nargs="+", action="store",
                                 help="Load embedding from directory or file path.")
        self.parser.add_argument("--embedding_col_names", action="store_true",
                                 help="Whether embedding's first row is names.")
        self.parser.add_argument("--label", action="store",
                                 help="Path to pre-classified labels.")
        self.parser.add_argument("--label_col_names", action="store_true",
                                 help="Whether embedding's first row is names.")
        self.parser.add_argument("--label_drop_col", type=int, nargs="+", action="store",
                                 help="Columns of label to be dropped.")
        self.parser.add_argument("--file_col_names", action="store_true",
                                 help="Whether file's first row is names.")
        self.parser.add_argument("--file_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--embedding_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--add_sample_index", action="store_true",
                                 help="Add sample index as first column of matrix.")
        
        # DR Evaluation
        self.parser.add_argument("--downsample", type=int, action="store",
                                 help="Downsample input file and embedding with n.")
        self.parser.add_argument("--k_fold", type=int, action="store",
                                 help="Repeatedly downsample k times to evaluate results.")
        
        # Dimension Reduction Parameters
        self.parser.add_argument("--out_dims", type=int, action="store",
                                 help="Output dimension")
        self.parser.add_argument("--perp", type=int, nargs="+", action="store",
                                 help="Perplexity or perplexity list for t-SNE.")
        self.parser.add_argument("--early_exaggeration", type=float, action="store",
                                 help="Early exaggeration factors for t-SNE.")
        self.parser.add_argument("--max_iter", type=int, action="store",
                                 help="Max iteration for t-SNE.")
        self.parser.add_argument("--early_exaggeration_iter", type=int, action="store",
                                 help="Iterations of early exaggeration.")
        self.parser.add_argument("--init", action="store",
                                 help="Initialization method for t-SNE.")
        self.parser.add_argument("--tsne_learning_rate", action="store",
                                 help="Learning rate for t-SNE.")
        self.parser.add_argument("--open_tsne_method", action="store",
                                 help="Method for openTSNE.")


    def parse(self, args: Optional[List[str]]=None) -> Dict[str, Optional[str]]:
        # Parse arguments

        arguments = self.parser.parse_args(args)
        arguments = vars(arguments)
        
        arguments["out"] = self.new_dir(arguments["out"], no_new_dir=arguments["no_new_dir"])
        arguments["out_dims"] = 2 if arguments["out_dims"] is None else arguments["out_dims"]
        arguments["delim"] = "\t" if arguments["delim"] is None else arguments["delim"]
        arguments["perp"] = 30 if arguments["perp"] is None else arguments["perp"]
        arguments["early_exaggeration"] = 12.0 if arguments["early_exaggeration"] is None else float(arguments["early_exaggeration"])
        arguments["max_iter"] = 1000 if arguments["max_iter"] is None else arguments["max_iter"]
        arguments["init"] = "random" if arguments["init"] is None else arguments["init"].lower()
        arguments["tsne_learning_rate"] = 200 if arguments["tsne_learning_rate"] is None else arguments["tsne_learning_rate"]
        arguments["open_tsne_method"] = "fft" if arguments["open_tsne_method"] is None else arguments["open_tsne_method"].lower()
        arguments["early_exaggeration_iter"] = 250 if arguments["early_exaggeration_iter"] is None else arguments["early_exaggeration_iter"]

        return arguments
    
    
    def new_dir(self, path:str, no_new_dir:bool):
        if no_new_dir:
            return path
        else:
            path = FileIO.make_dir(path)
            return path


if __name__ == "__main__":
    cmdargs = _Arguments().parse()
    main(cmdargs=cmdargs)