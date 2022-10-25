import numpy as np
from tuning.hyper_parameter_tuning_xgb import xgb
from tuning.plotting import plot_data, heat_map
from utilities import process_data
import json
import time


root_file = (
    "/Users/spencerhirsch/Documents/research/root_files/MZD_200_ALL/MZD_200_55.root"
)
root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/MZD_200_55_pd_model"


'''
    Function that deals with processing data. Utilizes the process_data class from the utilities 
    directory. Returns a final array with all of the values necessary for generating the models.
'''


def process():
    data = process_data("mc")  # declare data processing object
    data.extract_data(root_file, root_dir)
    data.prelim_cuts()
    data.dRgenCalc()
    data.SSM()

    d_r_cut = 0.05
    cut = "all"  # cut type ("all" applies cuts to every observable)
    data.dRcut(d_r_cut, cut)

    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    final_array = data.fillAndSort(save=True)

    return final_array


'''
    Function handles data processing for building the various xgboost models for analysis. Stores the data in
    their respective files and directories. Calls all of the necessary functions and creates all objects
    used for analysis. Driver function to collect all data for further analysis. 
'''


def xgmain():
    start = time.time()
    final_array = process()
    boost = xgb("mc")
    boost.split(final_array)

    '''
        Arrays to store the values of the hyperparameters that are run with the model. Need to look into
        using a more efficient testing algorithm. Bayesian hyperparameter tuning?
        
        Removed gblinear for booster parameter, inefficient and complications with xgboost, data for 
        default stored in archive file.
    '''

    booster_list = ['gbtree', 'dart'] # Determined gbtree is best
    alpha_array = [0, 1, 2, 3, 4, 5]   # L1 Regularization
    lambda_array = [0, 1, 2, 3, 4, 5]   # L2 Regularization
    eta_array = [0.6, 0.5, 0.4, 0.3, 0.1]    # Learning rate
    max_depth_array = [3, 6, 10, 12, 15]   # Maximum depth of the tree

    '''
        Iterate through all of the hyper parameters. Currently looking into the learning rate (eta) and the max
        depth of the tree.
    '''

    for val_alpha in alpha_array:
        for val_lambda in lambda_array:
            for val_eta in eta_array:
                for val_max_depth in max_depth_array:
                    _ = boost.xgb(single_pair=True, ret=True, eta=val_eta, max_depth=val_max_depth,
                                  reg_lambda=val_lambda, reg_alpha=val_alpha)
    '''
        Iterating over only the values of learning rate and the maximum depth of the tree.
        This data has been collected and logged in the archive file in the external SSD.
    '''
    # for val1 in eta_array:
    #     for val2 in max_depth_array:
    #         _ = boost.xgb(single_pair=True, ret=True, eta=val1, max_depth=val2)

    boost.model_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)

    obj_list = []
    for val in boost.model_list:
        obj_list.append(val.get_model())

    print("Completed.")

    class_out = resultDir + "/model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)

    end = time.time()
    total = end - start

    class_out = resultDir + "/time.json"
    out_file = open(class_out, "w")
    json.dump(total, out_file)
    print(total)


'''
    Driver function that handles each respective function. Takes input from the standard input stream
    and calls each function for its desired ability specified by the user. Makes the code much cleaner 
    and more concise.
'''

def main():
    print('Which program would you like to use:\n(1): Generate permutations of models \n(2): Generate Heat Map '
          '\n(3): Plot data')

    choice = input('Choice: ')

    if choice == '1':
        xgmain()
    elif choice == '2':
        print("Which metric would you like to plot?\n (1): time\n (2): mcc\n (3): f1")
        input_val = input('Choice: ')
        metric = ''

        if input_val == '1':
            metric = 'time'
        elif input_val == '2':
            metric = 'mcc'
        elif input_val == '3':
            metric = 'f1'
        else:
            print('Invalid input.')

        heat_map(metric)
    elif choice == '3':
        plot_data()
    else:
        print("Input invalid.")


main()

