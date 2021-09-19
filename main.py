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
    
    transform: Optional["np.ndarray"] = None    
    if cmdargs["transform"] is not None:
        transform = FileIO.load_data(files=cmdargs["transform"],
                                     col_names=cmdargs["file_col_names"],
                                     add_sample_index=cmdargs["add_sample_index"],
                                     drop_columns=cmdargs["file_drop_col"],
                                     delim=cmdargs["delim"])[1]
    
    if cmdargs["build_annoy"]:
        model: "AnnoyIndex" = Annoy.build_annoy(data=data[1])
        annoy_path: str = cmdargs["out"] + "/annoy.ann"
        Annoy.save_annoy(model, annoy_path)
    
    if cmdargs["downsample"] is not None:
        data_col_names: Optional["np.ndarray"] = data[0] if cmdargs["downsample_save_data_colnames"] is not None else None
        DownSample.downsample_from_data(data=data[1],
                                        n=cmdargs["downsample"],
                                        n_fold=cmdargs["k_fold"],
                                        out=cmdargs["out"],
                                        index_out=cmdargs["save_downsample_index"],
                                        replace=cmdargs["downsample_replace"],
                                        data_col_names=data_col_names)
    
    if cmdargs["evaluate"]:
        embedding: Optional[List["np.ndarray"]]
        labels: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        embedding_labels: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        comparison_file: Optional[List["np.ndarray"]] = None
        comparison_labels: Optional[Union[List["np.ndarray"], "np.ndarray"]] = None
        index: Optional[List["np.ndarray"]] = None
        
        embedding = FileIO.load_data(files=cmdargs["embedding"],
                                     col_names=cmdargs["embedding_col_names"],
                                     concat=cmdargs["concat"],
                                     add_sample_index=cmdargs["add_sample_index"],
                                     drop_columns=cmdargs["embedding_drop_col"])
        embedding.pop(0)
        
        if cmdargs["labels"] is not None:
            labels = FileIO.load_data(files=cmdargs["labels"],
                                     col_names=cmdargs["labels_col_names"],
                                     concat=cmdargs["concat"],
                                     add_sample_index=cmdargs["add_sample_index"],
                                     drop_columns=cmdargs["labels_drop_col"],
                                     dtype=int)[1]
            
        if cmdargs["embedding_labels"] is not None:
            embedding_labels = FileIO.load_data(files=cmdargs["embedding_labels"],
                                                col_names=cmdargs["labels_col_names"],
                                                concat=cmdargs["concat"],
                                                add_sample_index=cmdargs["add_sample_index"],
                                                drop_columns=cmdargs["labels_drop_col"],
                                                dtype=str)[1]
            
        if cmdargs["comparison_file"] is not None:
            comparison_file = FileIO.load_data(files=cmdargs["comparison_file"],
                                               col_names=cmdargs["comparison_file_col_names"],
                                               concat=False,
                                               add_sample_index=False,
                                               drop_columns=cmdargs["comparison_file_drop_col"])
            comparison_labels = FileIO.load_data(files=cmdargs["comparison_labels"],
                                                 col_names=cmdargs["comparison_labels_col_names"],
                                                 concat=False,
                                                 add_sample_index=False,
                                                 drop_columns=cmdargs["comparison_labels_drop_col"],
                                                 dtype=str)
            
            comparison_file.pop(0)
            comparison_labels.pop(0)
           
        results: List[List[Union[str, float]]]
        embedding_names: Optional["np.ndarray"] = None
        
        if len(cmdargs["embedding"]) == 1:
            embedding_names = np.array(FileIO._check_dir(cmdargs["embedding"][0]))
        else:
            embedding_names = np.array(cmdargs["embedding"])
                      
        if cmdargs["downsample_index_files"] is None:
            results= metric.Metric.run_metrics(data=data[1],
                                               embedding=embedding,
                                               methods=cmdargs["methods"],
                                               dist_metric=cmdargs["eval_dist_metric"],
                                               labels=labels,
                                               labels_embedding=embedding_labels,
                                               comparison_file=comparison_file,
                                               comparison_labels=comparison_labels,
                                               comparison_classes=cmdargs["comparison_classes"],
                                               embedding_names=embedding_names,
                                               data_annoy_path=cmdargs["file_annoy"],
                                               k=cmdargs["eval_k_neighbors"])
            
        else:
            index = FileIO.load_data(files=cmdargs["downsample_index_files"],
                                     col_names=False,
                                     concat=False,
                                     add_sample_index=False,
                                     drop_columns=None,
                                     dtype=int)
            index.pop(0)
            results = metric.Metric.run_metrics_downsample(data=data[1],
                                                           embedding=embedding,
                                                           methods=cmdargs["methods"],
                                                           dist_metric=cmdargs["eval_dist_metric"],
                                                           labels=labels,
                                                           labels_embedding=embedding_labels,
                                                           comparison_file=comparison_file,
                                                           comparison_labels=comparison_labels,
                                                           comparison_classes=cmdargs["comparison_classes"],
                                                           embedding_names=embedding_names,
                                                           downsample_indices=index,
                                                           data_annoy_path=cmdargs["file_annoy"],
                                                           k=cmdargs["eval_k_neighbors"])
            
        FileIO.save_list_to_csv(results, cmdargs["out"], "eval_metric")
        
    if cmdargs["split_train"] is not None:
        if cmdargs["split_train"] > 1 or cmdargs["split_train"]<0:
            raise ValueError("Training data percentage has to be between 0 and 1.")
        DownSample.train_test_split(data=data[1],
                                    train_percent=cmdargs["split_train"],
                                    col_names=data[0],
                                    out=cmdargs["out"])
        
    if cmdargs["cluster"]:
        cluster.Cluster.cluster(data[1],
                                methods=cmdargs["methods"],
                                out=cmdargs["out"])
    
    if cmdargs["dr"]:
        dr.DR.run_methods(data=data[1],
                          out=cmdargs["out"],
                          methods=cmdargs["methods"],
                          transform=transform,
                          out_dims=cmdargs["out_dims"],
                          save_embedding_colnames=cmdargs["save_embedding_colnames"],
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
                          SAUCIE_learning_rate=cmdargs["SAUCIE_learning_rate"],
                          kernel=cmdargs["kernelPCA"])
        
        
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
        self.parser.add_argument("--split_train", type=float, action="store",
                                 help="Train-test split of input data and save to output directory.")
        self.parser.add_argument("--downsample", type=int, action="store",
                                 help="Downsample input file and embedding with n.")

        # Methods: For all modes
        self.parser.add_argument("-m", "--methods", nargs="+", action="store",
                                 help="Methods to run: applies to all modules.")
        
        # File IO: Input Files
        self.parser.add_argument("-f", "--file", nargs="+", action="store",
                                 help="Path to directory or original files.")
        self.parser.add_argument("--transform", nargs="+", action="store",
                                 help="Path to directory or files to transform. The format must be the same as the input files from ``-f``.")
        self.parser.add_argument("--concat", action="store_true",
                                 help="Concatenate files, embeddings, and labels read.")
        self.parser.add_argument("--delim", type=str, action="store", default="\t",
                                 help="File delimiter.")
        self.parser.add_argument("--file_col_names", action="store_true",
                                 help="Whether file's first row is names.")
        self.parser.add_argument("--file_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--add_sample_index", action="store_true",
                                 help="Add sample index as first column of matrix.")
        # File IO: Output
        self.parser.add_argument("-o", "--out", type=str, action="store",
                                 help="Directory name for saving results")
        self.parser.add_argument("--no_new_dir", action="store_true",
                                 help="Save results directly in -o without creating new directory.")
        self.parser.add_argument("--save_embedding_colnames", action="store_true",
                                 help="Save column names for embedding after DR.")
        # File IO: Embedding and labels
        self.parser.add_argument("--embedding", nargs="+", action="store",
                                 help="Load embedding from directory or file path.")
        self.parser.add_argument("--embedding_col_names", action="store_true",
                                 help="Whether embedding's first row is names.")
        self.parser.add_argument("--embedding_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--labels", action="store",
                                 help="Path to pre-classified labels.")
        self.parser.add_argument("--embedding_labels", action="store",
                                 help="Path to pre-classified labels of embedding.")
        self.parser.add_argument("--labels_col_names", action="store_true",
                                 help="Whether embedding's first row is names.")
        self.parser.add_argument("--labels_drop_col", type=int, nargs="+", action="store",
                                 help="Columns of label to be dropped.")
        # File IO: Comparison EMbedding and Labels
        self.parser.add_argument("--comparison_file", nargs="+", action="store",
                                 help="Load comparison file from directory or file path.")
        self.parser.add_argument("--comparison_file_col_names", action="store_true",
                                 help="Whether comparison file's first row is names.")
        self.parser.add_argument("--comparison_file_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading comparison file.")
        self.parser.add_argument("--comparison_labels", nargs="+", action="store",
                                 help="Path to pre-classified comparison labels of embedding.")
        self.parser.add_argument("--comparison_labels_col_names", action="store_true",
                                 help="Whether comparison embedding labels' first row is names.")
        self.parser.add_argument("--comparison_labels_drop_col", type=int, nargs="+", action="store",
                                 help="Columns of comparison label to be dropped.")
        self.parser.add_argument("--comparison_classes", type=str, nargs="+", action="store",
                                 help="Classes to use for comparison.")
        
        # Downsampling
        self.parser.add_argument("--k_fold", type=int, action="store", default=1,
                                 help="Repeatedly downsample k times to evaluate results.")
        self.parser.add_argument("--save_downsample_index", action="store_true",
                                 help="Save indicies in a subdirectory 'index'.")
        self.parser.add_argument("--downsample_replace", action="store_true",
                                 help="Downsample with replacement.")
        self.parser.add_argument("--downsample_save_data_colnames", action="store_true",
                                 help="Whether to save column names for downsampled data.")
        
        # DR Evalation
        self.parser.add_argument("--eval_k_neighbors", type=int, action="store", default=100,
                                 help="Number of neighbors for local structure preservation metrics.")
        self.parser.add_argument("--downsample_index_files",nargs="+", action="store",
                                 help="File paths or directory path to saved downsample indicies as tsv.")
        self.parser.add_argument("--file_annoy", type=str, action="store",
                                 help="Path to the file's Annoy model.")
        self.parser.add_argument("--eval_dist_metric", type=str, action="store", default="PCD",
                                 help="Path to the file's Annoy model.")
        
        # Dimension Reduction Parameters
        self.parser.add_argument("--out_dims", type=int, action="store", default=2,
                                 help="Output dimension")
        self.parser.add_argument("--perp", type=int, nargs="+", action="store", default=30,
                                 help="Perplexity or perplexity list for t-SNE.")
        self.parser.add_argument("--early_exaggeration", type=float, action="store", default=12.0,
                                 help="Early exaggeration factors for t-SNE.")
        self.parser.add_argument("--max_iter", type=int, action="store", default=1000,
                                 help="Max iteration for t-SNE.")
        self.parser.add_argument("--early_exaggeration_iter", type=int, action="store", default=250,
                                 help="Iterations of early exaggeration.")
        self.parser.add_argument("--init", action="store", default="random",
                                 help="Initialization method for t-SNE.")
        self.parser.add_argument("--tsne_learning_rate", action="store", default="auto",
                                 help="Learning rate for t-SNE.")
        self.parser.add_argument("--open_tsne_method", action="store", default="fft",
                                 help="Method for openTSNE.")
        self.parser.add_argument("--dist_metric", type=str, action="store", default="euclidean",
                                 help="Distance metric for applicable methods.")
        
        #UMAP
        self.parser.add_argument("--umap_min_dist", type=float, action="store", default=0.1,
                                 help="min_dist for UMAP.")
        self.parser.add_argument("--umap_neighbors", type=int, action="store", default=15,
                                 help="Number of neighbors for UMAP.")
        
        #SAUCIE
        self.parser.add_argument("--SAUCIE_lambda_c", type=float, action="store", default=0,
                                 help="Information dimension regularization for SAUCIE.")
        self.parser.add_argument("--SAUCIE_lambda_d", type=float, action="store", default=0,
                                 help="Intracluster distance regularization for SAUCIE.")
        self.parser.add_argument("--SAUCIE_learning_rate", type=float, action="store", default=0.001,
                                 help="Learning rate for SAUCIE.")
        self.parser.add_argument("--SAUCIE_steps", type=int, action="store", default=1000,
                                 help="Maximum iteration for SAUCIE.")
        self.parser.add_argument("--SAUCIE_batch_size", type=int, action="store", default=256,
                                 help="Batch size for SAUCIE.")
        
        # KernelPCA
        self.parser.add_argument("--kernelPCA", type=str, action="store", default="poly",
                                 help="Kernel for kernel PCA.")
        

    def parse(self, args: Optional[List[str]]=None) -> Dict[str, Optional[str]]:
        # Parse arguments
        arguments = self.parser.parse_args(args)
        arguments = vars(arguments)
        
        if arguments["out"] is not None:
            arguments["out"] = self.new_dir(arguments["out"], no_new_dir=arguments["no_new_dir"])
            
        if arguments["save_downsample_index"]:
            assert arguments["out"] is not None
            index_out: str = arguments["out"] + "/index"
            arguments["save_downsample_index"] = self.new_dir(index_out, False)
        
        arguments["tsne_learning_rate"] = arguments["tsne_learning_rate"].lower() 
        if arguments["tsne_learning_rate"] != "auto":
            arguments["tsne_learning_rate"] = int(arguments["tsne_learning_rate"])
            
        arguments["init"] = arguments["init"].lower()
        arguments["open_tsne_method"] = arguments["open_tsne_method"].lower()
        arguments["dist_metric"] = arguments["dist_metric"].lower()

        return arguments
    
    
    def new_dir(self, path:str, no_new_dir:bool):
        # Create new directory or return the current path str.
        if no_new_dir:
            return path
        else:
            path = FileIO.make_dir(path)
            return path


if __name__ == "__main__":
    cmdargs = _Arguments().parse()
    main(cmdargs=cmdargs)