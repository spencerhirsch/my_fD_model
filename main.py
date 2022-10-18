import numpy
import bkg_pairing as bkg
from utilities import process_data
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from MLtools.xgb_plusplus import xgb
import sys


root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"
root_file = "/Users/spencerhirsch/Documents/research/MZD_125_15.root"
parent_path = "/Users/spencerhirsch/Documents/research/my_fD_model/histograms"


def main():
    argument = sys.argv[1]
    if argument == "plot":
        execute_plot()
    elif argument == "ML":
        final_array = bkg.execute()
        process_xgboost(final_array)


"""
    Calls the xgb boost function that handles processing the data and training the model.
"""


def process_xgboost(final_array):
    xgboost = xgb("mc")
    xgboost.split(final_array)
    _ = xgboost.xgb(single_pair=True, ret=True)
    xgboost.plotMatch()
    del xgboost


"""
Function used to plot the data generated in the root files. Makes it easier to see rather than viewing the data
using ROOT and TBrowser. Data generated is only data that we care about.
"""


def execute_plot():
    prodat = process_data("mc")
    prodat.extract_data(root_file, root_dir)
    prodat.prelim_cuts()
    plot(np.array(prodat.genAMu_pt), "pt")
    plot(np.array(prodat.genAMu_eta), "eta")
    plot(np.array(prodat.genAMu_phi), "phi")


def plot(numpy_array, name):
    filename = name
    plt.title(filename)
    plt.grid()
    filename = filename + ".png"

    for i in range(len(numpy_array[0])):
        if i == 0:
            color = "r"
        elif i == 1:
            color = "g"
        elif i == 2:
            color = "y"
        else:
            color = "b"

        label = "muon " + str(i + 1)

        """
        New way of calculating the bin size for the histogram. Using a 
        """
        q1, q3 = np.percentile(numpy_array[:, i], [75, 25])
        iqr = q3 - q1

        bin_width = 2 * iqr * len(numpy_array[:, i]) ** (-1 / 3)

        bins = (
            float((numpy.amax(numpy_array[:, i]) - numpy.amin(numpy_array[:, i])))
            / bin_width
        )

        plt.hist(
            numpy_array[:, i],
            bins=abs(int(bins)),
            color=color,
            label=label,
            alpha=0.2,
            edgecolor=color,
        )

    plt.savefig(filename)
    plt.legend(loc="upper right")
    plt.xlabel("GeV")
    plt.ylabel("Number of Events")
    plt.show()


main()
