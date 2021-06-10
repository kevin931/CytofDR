import numpy as np
from fileio import FileIO

import sys
import argparse
from typing import Optional, List, Dict, Union, Any

def simulate_random(n:int, p:int, col_index:bool=True, row_index:bool=True) -> "np.ndarray":
    
    mat: "np.ndarray" = np.random.rand(n, p)
    
    if col_index:
        col: "np.ndarray" = np.arange(p).reshape(1, p)
        mat = np.concatenate((col, mat))
    
    if row_index:
        row: "np.ndarray" = np.arange(n+1).reshape(n+1, 1)
        mat = np.concatenate((row, mat), axis=1)
    
    return mat


def simulate_fcs_downsample(n: Union[int, List[int]],
                            files: Union[str, List[str]],
                            concat: bool,
                            file_col_names: bool,
                            out: str,
                            file_drop_col: Optional[Union[int, List[int]]]=None,
                            add_sample_index: bool=False,
                            col_index: bool=True
                            ) -> None:
    
    data: List["np.ndarray"] = FileIO.load_data(files,
                                                concat=concat,
                                                col_names=file_col_names,
                                                drop_columns=file_drop_col,
                                                add_sample_index=add_sample_index)
    names = data[0]
    exprs: "np.ndarray" = data[1]
    
    if not isinstance(n, list):
        n = [n]
        
    i: int
    for i in n:
        indices = np.random.choice(exprs.shape[0], i, replace=False)
        
        if (col_index):
            mat = np.concatenate((names.reshape(1, names.shape[0]), exprs[indices]))
        else:
            mat = exprs[indices]
            
        file_path: str = "{}/{}.txt".format(out, i)
        np.savetxt(file_path, mat, fmt='%s',  delimiter="\t")
        

class _Arguments():
    # This class parses command line arguments.

    def __init__(self) -> None:
        # New parser with the appropriate flags.

        self.parser = argparse.ArgumentParser(description="Command Line Arguments")
        self.parser.add_argument("-n", type=int, nargs="+", action="store",
                                 help="Shape of simulation data.")
        self.parser.add_argument("-p", type=int, nargs="+", action="store",
                                 help="Shape of simulation data.")
        self.parser.add_argument("-m", "--method", action="store",
                                 help="Methods to run.")
        self.parser.add_argument("-o", "--out", action="store",
                                 help="Directory name for saving simulation datasets.")
        self.parser.add_argument("--col_index", action="store_true",
                                 help="Path to save results (txt or csv).")
        self.parser.add_argument("--row_index", action="store_true",
                                 help="Path to save results (txt or csv).")
        self.parser.add_argument("-f", "--files", nargs="+", action="store",
                                 help="Path to save results (txt or csv).")
        self.parser.add_argument("--add_sample_index", action="store_true",
                                 help="Add sample index as first column of matrix.")
        self.parser.add_argument("--file_col_names", action="store_true",
                                 help="Whether file's first row is names.")
        self.parser.add_argument("--file_drop_col", type=int, nargs="+", action="store",
                                 help="Columns to drop while reading files.")
        self.parser.add_argument("--concat", action="store_true",
                                 help="Concatenate files.")
        
        
    def parse(self, args: Optional[List[str]]=None) -> Dict[str, Any]:
        # Parse arguments

        arguments = self.parser.parse_args(args)
        arguments = vars(arguments)
        
        if arguments["n"] is None or arguments["out"] is None:
            print("'-n' and '-o' required. Exiting.")
            sys.exit()
        
        arguments["method"] = "random" if arguments["method"] is None else arguments["method"].lower()
        
        if arguments["p"] is not None and len(arguments["p"]) != len(arguments["n"]):
            print("'-n' and '-p' must have the same dimension.")
            sys.exit()
        
        return arguments
    
    
if __name__ == "__main__":
    
    cmdargs: Dict[str, Any] = _Arguments().parse()
    
    if cmdargs["method"] == "random":
        mat = simulate_random(n=cmdargs["n"][0], #TODO: Fix list
                              p=cmdargs["p"][0],
                              col_index=cmdargs["col_index"],
                              row_index=cmdargs["row_index"])

        file_path: str = "{}/{}_{}.txt".format(cmdargs["out"], cmdargs["n"][0], cmdargs["p"][0])
        np.savetxt(file_path, mat, delimiter="\t")
        
    if  cmdargs["method"] == "fcs_downsample":
        simulate_fcs_downsample(n=cmdargs["n"], 
                                files=cmdargs["files"], 
                                out=cmdargs["out"],
                                concat=cmdargs["concat"],
                                file_col_names=cmdargs["file_col_names"],
                                add_sample_index=cmdargs["add_sample_index"],
                                file_drop_col=cmdargs["file_drop_col"],
                                col_index=cmdargs["col_index"])