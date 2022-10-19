from MLtools.xgb_plusplus import xgb
import MLtools.xgb_plusplus
from utilities import process_data
import json

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

    # Default parameters, should include if None set to default, xgb_plus_plus accounts for this
    _ = boost.xgb(single_pair=True, ret=True)

    '''
        Iterate through all of the hyper parameters. Need to include the parameters that test
        the different boosters.
    '''

    for val in booster_list:
        _ = boost.xgb(single_pair=True, ret=True, booster=val)

    for val in eta_array:
        _ = boost.xgb(single_pair=True, ret=True, eta=val)

    for val in max_depth_array:
        _ = boost.xgb(single_pair=True, ret=True, eta=None, max_depth=val)

    for val1 in eta_array:
        for val2 in max_depth_array:
            _ = boost.xgb(single_pair=True, ret=True, eta=val1, max_depth=val2)

    xgbplus = MLtools.xgb_plusplus
    mod_list = xgbplus.model_list

    mod_list.sort(key=lambda x: (x.mcc, x.accuracy), reverse=True)
    obj_list = []

    for val in mod_list:
        obj_list = val.get_model()

    print("Completed.")

    class_out = resultDir + "/model_list.json"
    out_file = open(class_out, "w")
    json.dump(obj_list, out_file, indent=4)


xgmain()

