import pprint

import numpy as np

from tuning.hyper_parameter_tuning_xgb import xgb
from utilities import process_data
import json
import time
import matplotlib.pyplot as plt
import os

root_file = (
    "/Users/spencerhirsch/Documents/research/root_files/MZD_200_ALL/MZD_200_55.root"
)
root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/MZD_200_55_pd_model"


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

    booster_list = ['gbtree', 'dart']
    eta_array = [0.4, 0.3, 0.1, 0.01, 0.001, 0.0001]
    max_depth_array = [3, 6, 10, 20, 30, 50, 75, 100]

    '''
        Iterate through all of the hyper parameters. Need to include the parameters that test
        the different boosters.
    '''

    # for val in booster_list:
    #     _ = boost.xgb(single_pair=True, ret=True, booster=val)
    # Run with default booster, gbtree
    for val1 in eta_array:
        for val2 in max_depth_array:
            _ = boost.xgb(single_pair=True, ret=True, eta=val1, max_depth=val2)

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
        Plotting for various hyper-parameters in the model to take care of comparisons.
    '''
def plot_data():
    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_1021_647PM/model_list.json'
    f = open(dir)
    data = json.load(f)

    eta_array = [0.4, 0.3, 0.1, 0.01, 0.001, 0.0001]
    max_depth_array = [3, 6, 10, 20, 30, 50, 75, 100]

    data = sorted(data, key=lambda x: x['eta'])

    # pprint.pprint(sorted(data, key=lambda x: x['eta']))

    index = 0
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    for i in range(len(max_depth_array)):
        storage = []
        for val in data:
            if val['max depth'] == max_depth_array[index]:
                storage.append(val)

        eta = []
        max_depth = []
        f1 = []
        precision = []
        mcc = []
        accuracy = []
        for val in storage:
            eta.append(val['eta'])
            max_depth.append(val['max depth'])
            f1.append(val['f1'])
            precision.append(val['precision'])
            mcc.append(val['mcc'])
            accuracy.append(val['accuracy'])

        fig, ax = plt.subplots()
        plt.title('Fixed Max Depth of %s' % max_depth_array[index], fontsize=15)
        default_x_ticks = range(len(eta))
        plt.xlabel('Learning Rate (eta)', fontsize=10, loc='right')
        plt.ylabel('Value', fontsize=10, loc='top')
        ax.grid()
        ax.plot(eta, f1, label='f1 Score', marker='D', linewidth=1)
        ax.plot(eta, mcc, label='mcc', marker='D', linewidth=1)
        ax.plot(eta, precision, label='precision', marker='D', linewidth=1)
        ax.plot(eta, accuracy, label='accuracy', marker='D', linewidth=1)
        ax.legend(loc='lower right', prop={'size': 12})


        fig.canvas.draw()
        plt.show()
        path = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/plots'

        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        fig.savefig(path + '/%s_max_depth_comparison.pdf' % max_depth_array[index])

        index += 1


def plot_heat():
    dir = '/Volumes/SA Hirsch/Florida Tech/research/dataframes/archive/data_1021_647PM/model_list.json'
    f = open(dir)
    data = json.load(f)

    max_depth_array = [3, 6, 10, 20, 30, 50, 75, 100]
    eta_array = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.4]
    mcc = []

    data = sorted(data, key=lambda x: x['eta'], reverse=False)
    index = 0
    for i in range(len(max_depth_array)):
        storage = []
        mmcc = []
        data = sorted(data, key=lambda x: x['eta'])
        for val in data:
            if val['max depth'] == max_depth_array[index]:
                storage.append(val)

        for val in storage:
            mmcc.append(val['mcc'])
        print(storage)
        print()
        mcc.append(mmcc)
        index += 1

    mcc.reverse()
    mcc = np.array(mcc)

    print(mcc)
    plt.rcParams.update({'font.size': 12})  # Increase font size for plotting
    fig, ax = plt.subplots()
    im = ax.imshow(mcc)
    # mesh = ax.pcolormesh(mcc, max_depth_array, eta_array, edgecolors='k', linewidths=0.5, cmap='inferno')
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar = plt.colorbar(mesh)
    cbar.ax.set_ylabel('mcc', rotation=-90, va="bottom")

    max_depth_array.reverse()
    ax.set_xticks(np.arange(len(eta_array)), labels=eta_array)
    ax.set_yticks(np.arange(len(max_depth_array)), labels=max_depth_array)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(max_depth_array)):
        for j in range(len(eta_array)):
            text = ax.text(j, i, str(mcc[i, j])[:5],
                           ha="center", va="center", color="w", fontsize=8)

    ax.set_title("Matthew's Correlation Coefficient")
    fig.tight_layout()
    plt.show()

plot_heat()
# plot_data()
# xgmain()
