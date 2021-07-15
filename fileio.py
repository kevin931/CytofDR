import numpy as np

import _csv
import csv
import os
from typing import Union, Optional, Any, List


class FileIO():
    
    @staticmethod
    def load_data(files: Union[List[str], str],
                  concat: bool=False,
                  col_names: bool=True,
                  add_sample_index: bool=True,
                  drop_columns: Optional[Union[int, List[int]]]=None,
                  delim: str="\t",
                  dtype = float
                  ) -> List["np.ndarray"]:
    
        exprs: Optional["np.ndarray"] = None
        return_files: List["np.ndarray"] = []
        
        if not isinstance(files, list):
            files = FileIO._check_dir(files)
        elif len(files)==1:
            files = FileIO._check_dir(files[0])
            
        skiprows: int=0
        
        for i, file in enumerate(files):
            # Load column names
            if i==0:
                if col_names:
                    names: "np.ndarray" = np.loadtxt(fname=file, dtype ="str", max_rows=1, delimiter=delim)
                    if add_sample_index:
                        names = np.concatenate((np.array(["index"]), names))
                    if drop_columns is not None:
                        names = np.delete(names, drop_columns)
                    return_files.append(names)
                    skiprows = 1
                    print(names)
                else:
                    return_files.append(np.array(None))
            
            # Load Data and add sample index
            f: "np.ndarray" = np.loadtxt(fname=file, dtype=dtype, skiprows=skiprows, delimiter=delim)
            
            if add_sample_index:
                index: "np.ndarray" = np.repeat(i, f.shape[0]).reshape(f.shape[0],1)
                f = np.column_stack((index, f))
                
            if drop_columns is not None:
                f = np.delete(f, drop_columns, axis=1)
            
            # Concatenate
            if concat:
                if i==0:
                    exprs = f
                else:
                    exprs = np.concatenate((exprs, f))
            else:
                return_files.append(f)
                
        if exprs is not None:
            return_files.append(exprs)
            
        return return_files
    
    
    @staticmethod
    def _check_dir(path: str) -> List[str]:
        if os.path.isdir(path):
            files: List[str] = os.listdir(path)
            files = [path+"/"+f for f in files if os.path.isfile(f)]
            return files
        else:
            return [path]
    
    
    @staticmethod
    def save_list_to_csv(data: List[List[Any]], dir_path: str, file_name:str):
    
        file_path: str = "{}/{}.csv".format(dir_path, file_name)
        
        w: "_csv._writer" = csv.writer(open(file_path, "w"))
        
        i: int
        j: int    
        for i in range(len(data[0])):
            row: List[Any] = []
            for j in range(len(data)):
                row.append(data[j][i])
            w.writerow(row)
            
            
    @staticmethod
    def save_np_array(array: "np.ndarray",
                      dir_path: str,
                      file_name: str,
                      col_names: Optional["np.ndarray"]=None,
                      auto_add_extension: bool=True,
                      dtype: str="%.18e") -> None:
    
        save_path: str = "{}/{}".format(dir_path, file_name)
        if auto_add_extension:
            save_path = save_path + ".txt"
        print("Saving file to: " + save_path)
        
        with open(save_path, "w") as f:
            if col_names is not None:
                f.write("\t".join(list(col_names)))
                f.write("\n")
            np.savetxt(f, array, delimiter="\t", fmt=dtype)
            
    
    @staticmethod
    def make_dir(dir_path: str, counter: int=0):
        
        dir_path = dir_path.rstrip("/")
        if counter==0:
            new_dir_path = dir_path
        else:
            new_dir_path = dir_path + str(counter)
        
        try:
            os.makedirs(new_dir_path)
        except FileExistsError:
            new_dir_path = FileIO.make_dir(dir_path, counter+1)
            
        return new_dir_path