def execute():
    # Constants
    # Original
    # root_file = 'ROOT_Files/MZD_200_ALL/MZD_200_55.root'
    # My file
    root_file = "/Users/spencerhirsch/Documents/research/MZD_125_15.root"

    root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
    dRcut = 0.05
    cut = "all"  # cut type ("all" applies cuts to every observable)
    from utilities import process_data

    data = process_data("mc")  # declare data processing object
    data.extract_data(root_file, root_dir)
    data.prelim_cuts()
    # data.removeBadEvents("all")
    data.dRgenCalc()
    data.SSM()
    data.dRcut(dRcut, cut)
    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    data.permutations()
    data.invMassCalc()
    data.dR_diMu()
    final_array = data.fillAndSort(save=True)
    # final_array = data.fillFinalArray()
    return final_array
