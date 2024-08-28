#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
import numpy as np
#import matplotlib.pyplot as plt
import itertools
import copy

def create_combined_param_dict(param_dict):
    """
    This function takes a dictionary of the varying parameters with corresponding
    values of each parameters and returns a list of dictionaries. Each dictionary
    has the same structure as a set of options passed to the model file during 
    training
    ----
    - param_dict: A dictionary such as:
                {
                 "n_hidden":[20, 30, 40, 50, 60],
                 "lr": [1e-3, 1e-4],
                 "num_epochs":[2000, 5000],
                 "num_layers":[1,2]
                }
    Returns:
    - list_of_param_combinations:
        [
        {"num_hidden":20, "lr":1e-3, "num_epochs":2000, "num_layers":1, ...},
        {"num_hidden":20, "lr":1e-4, "num_epochs":2000, "num_layers":1, ...},
        {"num_hidden":30, "lr":1e-3, "num_epochs":2000, "num_layers":1, ...},
        {"num_hidden":30, "lr":1e-4, "num_epochs":2000, "num_layers":1, ...},
        :
        ]
    """
    keys, values = zip(*param_dict.items())
    list_of_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return list_of_param_combinations

def create_list_of_dicts(options, model_type, param_dict):
    """ 
    This function creates a list of dicts using the function
    `create_combined_param_dict()`, and also adds the other
    'non-varying' options to create the full set of options to be 
    passed directly to the model
    ---
    - options: A dictionary containing options that are usually passed to the model
    - model_type: "lstm" / "gru" / "rnn"
    - param_dict: A dictionary of fields and corresponding values such as:
                {
                 "n_hidden":[20, 30, 40, 50, 60],
                 "lr": [1e-3, 1e-4],
                 "num_epochs":[2000, 5000],
                 "num_layers":[1,2]
                }
    Returns:
        params_dict_list_all : List of param combinations (including the non-varying parameters)
    """
    params_dict_list_all = []
    param_combinations_dict = create_combined_param_dict(param_dict)
    for p_dict in param_combinations_dict:
        keys = p_dict.keys()
        tmp_dict = copy.deepcopy(options)
        for key in keys:
            tmp_dict['rnn_params_dict'][model_type][key] = p_dict[key]
        params_dict_list_all.append(copy.deepcopy(tmp_dict))
        
    #print("Grid-search will be computed for the following set of parameter lists:\n{}".format(len(params_dict_list_all)))
    return params_dict_list_all
