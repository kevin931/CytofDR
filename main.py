import numpy as np
from annoy import AnnoyIndex
import dr
import metric
import cluster
from fileio import FileIO
from util import DownSample, Annoy

import argparse
from typing import Optional, Union, List, Dict, Any


def main(cmdargs: Dict[str, Any]):
    
    data: List["np.ndarray"] = FileIO.load_data(files=cmdargs["file"],
                                                col_names=cmdargs["file_col_names"],
                                                add_sample_index=cmdargs["add_sample_index"],
                                                drop_columns=cmdargs["file_drop_col"],
                                                delim=cmdargs["delim"])
    
    if cmdargs["build_annoy"]:
        model: "AnnoyIndex" = Annoy.build_annoy(data=data[1])
        annoy_path: str = cmdargs["out"] + "/annoy.ann"
        Annoy.save_annoy(model, annoy_path)
    
    if cmdargs["embedding"] is None and cmdargs["downsample"] is not None:
        DownSample.downsample_from_data(data=data[1],
                                        n=cmdargs["downsample"],
                                        n_fold=cmdargs["k_fold"],
                                        save_downsample_index=cmdargs["save_downsample_index"])
    
    if cmdargs["evaluate"]:
        embedding: Optional[List["np.ndarray"]]
        label: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        label_embedding: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        index: Optional[List["np.ndarray"]] = None
        
        embedding = FileIO.load_data(files=cmdargs["embedding"],
                                        col_names=cmdargs["embedding_col_names"],
                                        concat=cmdargs["concat"],
                                        add_sample_index=cmdargs["add_sample_index"],
                                        drop_columns=cmdargs["embedding_drop_col"])
        embedding.pop(0)
        
        if cmdargs["label"] is not None:
            label = FileIO.load_data(files=cmdargs["label"],
                                     col_names=cmdargs["label_col_names"],
                                     concat=cmdargs["concat"],
                                     add_sample_index=cmdargs["add_sample_index"],
                                     drop_columns=cmdargs["label_drop_col"],
                                     dtype=int)[1]
            
        if cmdargs["label_embedding"] is not None:
            label_embedding = FileIO.load_data(files=cmdargs["label_embedding"],
                                               col_names=cmdargs["label_col_names"],
                                               concat=cmdargs["concat"],
                                               add_sample_index=cmdargs["add_sample_index"],
                                               drop_columns=cmdargs["label_drop_col"])[1]
            
        if cmdargs["downsample_index_files"] is not None:
            index = FileIO.load_data(files=cmdargs["downsample_index_files"],
                                     col_names=False,
                                     concat=False,
                                     add_sample_index=False,
                                     drop_columns=None,
                                     dtype=int)
            index.pop(0)
           
        results: List[List[Union[str, float]]]
        embedding_names: Optional["np.ndarray"] = None
        
        if len(cmdargs["embedding"]) == 1:
            embedding_names = np.array(FileIO._check_dir(cmdargs["embedding"][0]))
        else:
            embedding_names = np.array(cmdargs["embedding"])
                      
        if cmdargs["downsample"] is None and cmdargs["downsample_index_files"] is None:
            results= metric.Metric.run_metrics(data=data[1],
                                               embedding=embedding,
                                               methods=cmdargs["methods"],
                                               labels=label,
                                               labels_embedding=label_embedding,
                                               embedding_names=embedding_names, 
                                               data_annoy_path=cmdargs["file_annoy"],
                                               k=cmdargs["eval_k_neighbors"])
            
        else:
            results = metric.Metric.run_metrics_downsample(data=data[1],
                                                           embedding=embedding,
                                                           methods=cmdargs["methods"],
                                                           labels=label,
                                                           labels_embedding=label_embedding,
                                                           embedding_names=embedding_names,
                                                           downsample=cmdargs["downsample"],
                                                           n_fold=cmdargs["k_fold"],
                                                           downsample_indices=index,
                                                           save_indices_dir=cmdargs["save_downsample_index"], 
                                                           data_annoy_path=cmdargs["file_annoy"],
                                                           k=cmdargs["eval_k_neighbors"])
            
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
                          open_tsne_method=cmdargs["open_tsne_method"],
                          dist_metric=cmdargs["dist_metric"],
                          umap_min_dist=cmdargs["umap_min_dist"],
                          umap_neighbors=cmdargs["umap_neighbors"],
                          SAUCIE_lambda_c=cmdargs["SAUCIE_lambda_c"],
                          SAUCIE_lambda_d=cmdargs["SAUCIE_lambda_d"],
                          SAUCIE_steps=cmdargs["SAUCIE_steps"],
                          SAUCIE_batch_size=cmdargs["SAUCIE_batch_size"],
                          SAUCIE_learning_rate=cmdargs["SAUCIE_learning_rate"])
        
        
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
        self.parser.add_argument("--build_annoy", action="store_true",
                                 help="Build Annoy model.")   

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
        self.parser.add_argument("--label_embedding", action="store",
                                 help="Path to pre-classified labels of embedding.")
        self.parser.add_argument("--label_col_names", action="store_true",
                                 help="Whether embedding's first row is names.")
        self.parser.add_argument("--label_drop_col", type=int, nargs="+", action="store",
                                 help="Columns of label to be dropped.")
        self.parser.add_argument("--file_col_names", action="store_true",
                                 help="Whether file's first row is names.")
        self.parser.add_argument("--file_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--file_annoy", type=str, action="store",
                                 help="Path to the file's Annoy model.")
        self.parser.add_argument("--embedding_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--add_sample_index", action="store_true",
                                 help="Add sample index as first column of matrix.")
        
        # DR Evaluation
        self.parser.add_argument("--downsample", type=int, action="store",
                                 help="Downsample input file and embedding with n.")
        self.parser.add_argument("--k_fold", type=int, action="store",
                                 help="Repeatedly downsample k times to evaluate results.")
        self.parser.add_argument("--save_downsample_index", action="store",
                                 help="Directory path to save indicies used for downsampling.")
        self.parser.add_argument("--downsample_index_files",nargs="+", action="store",
                                 help="File paths or directory path to saved downsample indicies as tsv.")
        self.parser.add_argument("--eval_k_neighbors", type=int, action="store",
                                 help="Number of neighbors for local structure preservation metrics.")
        
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
        self.parser.add_argument("--dist_metric", type=str, action="store",
                                 help="Distance metric for applicable methods.")
        
        #UMAP
        self.parser.add_argument("--umap_min_dist", type=float, action="store",
                                 help="min_dist for UMAP.")
        self.parser.add_argument("--umap_neighbors", type=int, action="store",
                                 help="Number of neighbors for UMAP.")
        
        #SAUCIE
        self.parser.add_argument("--SAUCIE_lambda_c", type=float, action="store",
                                 help="Information dimension regularization for SAUCIE.")
        self.parser.add_argument("--SAUCIE_lambda_d", type=float, action="store",
                                 help="Intracluster distance regularization for SAUCIE.")
        self.parser.add_argument("--SAUCIE_learning_rate", type=float, action="store",
                                 help="Learning rate for SAUCIE.")
        self.parser.add_argument("--SAUCIE_steps", type=int, action="store",
                                 help="IMaximum iteration for SAUCIE.")
        self.parser.add_argument("--SAUCIE_batch_size", type=int, action="store",
                                 help="Batch size for SAUCIE.")
        

    def parse(self, args: Optional[List[str]]=None) -> Dict[str, Optional[str]]:
        # Parse arguments

        arguments = self.parser.parse_args(args)
        arguments = vars(arguments)
        
        if arguments["out"] is not None:
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
        arguments["dist_metric"] = "euclidean" if arguments["dist_metric"] is None else arguments["dist_metric"]
        arguments["umap_min_dist"] = 0.1 if arguments["umap_min_dist"] is None else arguments["umap_min_dist"]
        arguments["umap_neighbors"] = 15 if arguments["umap_neighbors"] is None else arguments["umap_neighbors"]
        arguments["SAUCIE_lambda_c"] = 0 if arguments["SAUCIE_lambda_c"] is None else arguments["SAUCIE_lambda_c"]
        arguments["SAUCIE_lambda_d"] = 0 if arguments["SAUCIE_lambda_d"] is None else arguments["SAUCIE_lambda_d"]
        arguments["SAUCIE_learning_rate"] = 0.001 if arguments["SAUCIE_learning_rate"] is None else arguments["SAUCIE_learning_rate"]
        arguments["SAUCIE_batch_size"] = 256 if arguments["SAUCIE_batch_size"] is None else arguments["SAUCIE_batch_size"]
        arguments["SAUCIE_steps"] = 1000 if arguments["SAUCIE_steps"] is None else arguments["SAUCIE_steps"]

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