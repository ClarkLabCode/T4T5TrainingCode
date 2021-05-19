import scipy.io as sio
import numpy as np
import os
import glob

def load_all_params(path):
    run_results = []
    for file_idx,filename in enumerate(glob.iglob(os.path.join(path , "**/*.mat"), recursive=True)):
        run_results.append(sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['param_dict'])
    
    return run_results

def filter_and_sort_params(params,**kwargs):
    out_params = []
    for this_param in params:
        include_this_param = True
        for key, value in kwargs.items():
            if getattr(this_param,key) != value:
                include_this_param = False
                break
        if include_this_param:
            out_params.append(this_param)
    
    val_metric = []
    if hasattr(out_params[0], 'r2'):
        val_metric = [p.r2[-1] for p in out_params]
    elif hasattr(out_params[0], 'accuracy'):
        val_metric = [p.accuracy[-1] for p in out_params]
    
    new_order = np.flip(np.argsort(val_metric))

    return [out_params[i] for i in new_order]

def convert_tensor_to_plot_matrix(tensor):
    sub_matrix_size = np.sqrt(tensor.shape[0]).astype(int)
    matrix = np.empty((sub_matrix_size*tensor.shape[1], sub_matrix_size*tensor.shape[2]))
    for i in range(tensor.shape[2]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[0]):
                row = np.mod(k,sub_matrix_size) + sub_matrix_size*j
                col = k//sub_matrix_size + sub_matrix_size*i
                matrix[row,col] = tensor[k,j,i]
    return matrix