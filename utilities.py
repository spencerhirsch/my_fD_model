import numpy as np
import pandas as pd
from math import sqrt
from random import randint
from tqdm import tqdm  # Progress bar for loops
from uproot import open as openUp  # Transfer ROOT data into np arrays
import numpy as np
import sys
import os

sys.path.insert(0, "~/Documents/research/my_fD_model")
from file_utils import *

global resultDir
global cutflow_dir
global bad_phi
global bad_eta
global bad_pT
global dR_cut
global root_dir

resultDir = (
    "dataframes/"  # base dir for all other subdirs which contain info/plots/data
)
cutflow_dir = "cutFlowAnalyzerPXBL4PXFL3"  # the directory in the ROOT file which contains the tree of results
bad_phi = -100
bad_eta = 2.4
bad_pT = -100
dR_cut = 0.05  # threshold for dR between gen and sel muons (all dRs lower than 0.05 will remain)
root_dir = "cutFlowAnalyzerPXBL4PXFL3;1/Events;1"


class process_data:
    """
    Processes root files. Functions:
    __init__:      Class constructor.
    extract_data():  Opens the root file, extracts the observables of interest,
                   and assigns these as an attribute of the class.

    Applies preliminary cuts

    """

    def __init__(self, dataset, ml_met=False):
        """
        Constructor for the processData class.

        Positional argument:
        dataset:    (str) The type of dataset. Options:
                        "mc":     Monte-Carlo (MC) signal samples. Contains both
                                  gen- and sel-level information.
                        "bkg":    MC background samples.
                        "sig":    MC signal samples. Does not contain gen-
                                  level information.
                        "ntuple": Raw n-tuples that have't been run through the
                                  cutflow analysis code.

        Optional argument:
        ml_met:     (bool) Compute the machine learning model metrics (retrieve extra observables
                    from the dataset and store). Default is False.

        Change log:
        S.D.B., 2022/08/30
            - Added the optional argument ml_met.
        """

        # Check arguments
        if dataset not in ["mc", "bkg", "sig", "ntuple"]:
            print_error("Dataset type must be either 'mc', 'bkg', 'sig', or 'ntuple'.")
            sys.exit()
        else:
            self.dataset = dataset  # store dataset type as attribute

        if ml_met and self.dataset not in [
            "mc",
            "sig",
        ]:  # only mc/sig samples have generator level information
            self.ml_met = ml_met  # store ml_met bool as attribute
        elif ml_met and self.dataset not in ["mc", "sig"]:
            print_error(
                "Can only use 'mc' or 'sig' datasets when extracting and saving observables used in computing the machine learning metrics."
            )
            sys.exit()

    def extract_data(self, root_file, root_dir, ret=False, verbose=False):
        """
        Accepts absolute (or relative) path for a ROOT file and extracts all
        observables of interestevent data. These data are stored as attributes
        of the process_data class, and can optionally be returned.

        Positional arguments:
        root_file:  (str)  The absolute or relative path to the root file.

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:

        selMu_eta:     (np.array, N_evt X 4; float) The eta coordinate of the sel [reconstructed (reco)] level  of each muon for each event. (Optional.)
        selMu_phi:     (np.array, N_evt X 4; float) The phi coordinate of the sel (reco) level  of each muon for each event. (Optional.)
        selMu_pt:      (np.array, N_evt X 4; float) The transverse momentum, pT, value of the sel (reco) level of each muon for each event. (Optional.)
        selMu_charge:  (np.array, N_evt X 4; float) The charge of the sel (reco) level of each muon for each event. (Optional.)

        Only for 'mc' datasets (and 'sig' if ml_met = True):
            genAMu_eta:    (np.array, N_evt X 4; float) The eta coordinate of the gen level [generator (pre-recoonstruction)] of each muon for each event. (Optional.)
            genAMu_phi:    (np.array, N_evt X 4; float) The phi coordinate of the gen level  of each muon for each event. (Optional.)
            genAMu_pt:     (np.array, N_evt X 4; float) The transverse momentum, pT, value of the gen level of each muon for each event. (Optional.)
            genAMu_charge: (np.array, N_evt X 4; float) The charge of the gen level of each muon for each event. (Optional.)

        Change log:
        S.D.B., 2022/08/30
            - Added the printing to std. out in color via the print_alert() function in file_utils.
        """

        self.fileName = root_file.split("/")[-1]  # get file name

        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Extracting data from %s" % self.fileName)
            print_alert(60 * "*")
            print_alert("\n\n")

        if self.dataset == "ntuple":
            file = openUp(root_file)
            folder = file[
                cutflow_dir
            ]  # ntuples have the tree in a directory, so first get the contents of the directory
            data = folder["Events"]  # get tree and all branches/leaves
        else:
            file = openUp(root_file)
            data = file[root_dir]  # get tree and all branches/leaves

        if verbose:
            print_alert("Extracting sel mu data (eta, phi, pT, and charge)")

        # get reco level eta info for all events as np.array
        selMu0_eta = np.asarray(data["selMu0_eta"].array())
        selMu1_eta = np.asarray(data["selMu1_eta"].array())
        selMu2_eta = np.asarray(data["selMu2_eta"].array())
        selMu3_eta = np.asarray(data["selMu3_eta"].array())
        self.selMu_eta = np.column_stack(
            (selMu0_eta, selMu1_eta, selMu2_eta, selMu3_eta)
        )

        # get reco level phi info for all events as np.array
        selMu0_phi = np.asarray(data["selMu0_phi"].array())
        selMu1_phi = np.asarray(data["selMu1_phi"].array())
        selMu2_phi = np.asarray(data["selMu2_phi"].array())
        selMu3_phi = np.asarray(data["selMu3_phi"].array())
        self.selMu_phi = np.column_stack(
            (selMu0_phi, selMu1_phi, selMu2_phi, selMu3_phi)
        )

        # get reco level pT info for all events as np.array
        selMu0_pt = np.asarray(data["selMu0_pT"].array())
        selMu1_pt = np.asarray(data["selMu1_pT"].array())
        selMu2_pt = np.asarray(data["selMu2_pT"].array())
        selMu3_pt = np.asarray(data["selMu3_pT"].array())
        self.selMu_pt = np.column_stack((selMu0_pt, selMu1_pt, selMu2_pt, selMu3_pt))

        # get reco level charge info for all events as np.array
        selMu0_charge = np.asarray(data["selMu0_charge"].array())
        selMu1_charge = np.asarray(data["selMu1_charge"].array())
        selMu2_charge = np.asarray(data["selMu2_charge"].array())
        selMu3_charge = np.asarray(data["selMu3_charge"].array())
        self.selMu_charge = np.column_stack(
            (selMu0_charge, selMu1_charge, selMu2_charge, selMu3_charge)
        )

        if self.dataset == "bkg" or self.dataset == "ntuple":
            pass
        elif self.dataset == "mc" or self.ml_met:
            if verbose:
                print_alert("Extracting gen mu data (eta, phi, pT, and charge)")

            # get generator (gen) level eta info for all events as np.array
            genA0Mu0_eta = np.asarray(data["genA0Mu0_eta"].array())
            genA0Mu1_eta = np.asarray(data["genA0Mu1_eta"].array())
            genA1Mu0_eta = np.asarray(data["genA1Mu0_eta"].array())
            genA1Mu1_eta = np.asarray(data["genA1Mu1_eta"].array())
            self.genAMu_eta = np.column_stack(
                (genA0Mu0_eta, genA0Mu1_eta, genA1Mu0_eta, genA1Mu1_eta)
            )

            # get gen level phi info for all events as np.array
            genA0Mu0_phi = np.asarray(data["genA0Mu0_phi"].array())
            genA0Mu1_phi = np.asarray(data["genA0Mu1_phi"].array())
            genA1Mu0_phi = np.asarray(data["genA1Mu0_phi"].array())
            genA1Mu1_phi = np.asarray(data["genA1Mu1_phi"].array())
            self.genAMu_phi = np.column_stack(
                (genA0Mu0_phi, genA0Mu1_phi, genA1Mu0_phi, genA1Mu1_phi)
            )

            # get gen level pT info for all events as np.array
            genA0Mu0_pt = np.asarray(data["genA0Mu0_pt"].array())
            genA0Mu1_pt = np.asarray(data["genA0Mu1_pt"].array())
            genA1Mu0_pt = np.asarray(data["genA1Mu0_pt"].array())
            genA1Mu1_pt = np.asarray(data["genA1Mu1_pt"].array())
            self.genAMu_pt = np.column_stack(
                (genA0Mu0_pt, genA0Mu1_pt, genA1Mu0_pt, genA1Mu1_pt)
            )

            # get gen level charge info for all events as np.array
            genA0Mu0_charge = np.asarray(data["genA0Mu0_charge"].array())
            genA0Mu1_charge = np.asarray(data["genA0Mu1_charge"].array())
            genA1Mu0_charge = np.asarray(data["genA1Mu0_charge"].array())
            genA1Mu1_charge = np.asarray(data["genA1Mu1_charge"].array())
            self.genAMu_charge = np.column_stack(
                (genA0Mu0_charge, genA0Mu1_charge, genA1Mu0_charge, genA1Mu1_charge)
            )

        del file

        if ret:
            if self.dataset == "mc" or self.ml_met:
                if verbose:
                    print_alert(
                        "Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge"
                    )

                return (
                    self.selMu_eta,
                    self.selMu_phi,
                    self.selMu_pt,
                    self.selMu_charge,
                    self.genAMu_eta,
                    self.genAMu_phi,
                    self.genAMu_pt,
                    self.genAMu_charge,
                )
            elif (
                self.dataset == "bkg"
                or self.dataset == "sig"
                or self.dataset == "ntuple"
            ):
                if verbose:
                    print_alert(
                        "Arrays returned: selMu_eta, selMu_phi, selMu_pt, selMu_charge"
                    )

                return self.selMu_eta, self.selMu_phi, self.selMu_pt, self.selMu_charge

    def prelim_cuts(self, ret=False, verbose=False):
        """
        Applies preliminary cuts based on eta, phi, pT, and charge.
        Stephen D. Butalla & Mehdi Rahmani
        2022/08/30, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Change log:
        S.D.B., 2022/08/30
            - Consolidated the cut process from removeBadEvents() and removed this function
              from the class.
            - Added the printing to std. out in color via the print_alert() function in file_utils.
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Applying preliminary cuts...")
            print_alert(60 * "*")
            print_alert("\n\n")
            print_alert(
                "Determing the events that were not properly reconstructed or are outside of the geometric acceptance of the detector system (eta > 2.4)"
            )

        badPhi = np.unique(np.where(self.selMu_phi == bad_phi)[0])
        badSelEta = np.unique(np.where(abs(self.selMu_eta) > bad_eta)[0])
        badSelpT = np.unique(np.where(self.selMu_pt == bad_pT)[0])
        badSelCharge = np.unique(
            np.where(np.sum(self.selMu_charge, axis=1) != 0)
        )  # Remove event if the sum of the charge of all four muons isn't 0.

        if self.dataset == "mc" or self.ml_met:
            badGenAEta = np.unique(np.where(abs(self.genAMu_eta) > 2.4)[0])

        # Convert to lists so we can add without broadcasting problems
        badPhi = list(badPhi)
        badSelEta = list(badSelEta)
        badSelpT = list(badSelpT)
        badCharge = list(badSelCharge)

        if self.dataset == "mc" or self.ml_met:
            badGenAEta = list(badGenAEta)

        if (
            self.dataset == "bkg"
            or self.dataset == "sig"
            or self.dataset == "ntuple"
            and self.ml_met
        ):
            self.badEvents = sorted(
                np.unique(badPhi + badSelEta + badSelpT + badCharge)
            )
        else:
            self.badEvents = sorted(
                np.unique(badPhi + badGenAEta + badSelEta + badSelpT + badCharge)
            )  # Add lists, return unique values, and sort to preserve order

        if verbose:
            print_alert(25 * "*" + " CUT INFO " + 25 * "*" + "\n")
            print_alert(
                "Total number of events failing reconstruction in phi: %d" % len(badPhi)
            )
            print_alert(
                "Total number of events with sel eta > 2.4: %d" % len(badSelEta)
            )

            if self.dataset == "mc" or self.ml_met:
                print_alert(
                    "Total number of events with gen eta > 2.4: %d" % len(badGenAEta)
                )

            print_alert(
                "Total number of events with sel pT == -100: %d".format(len(badSelpT))
            )
            print_alert(
                "Total number of events failing charge reconstruction: {}".format(
                    len(badCharge)
                )
            )
            print_alert("Total number of bad events: {}".format(len(self.badEvents)))
            print_alert("\n" + 60 * "*")

        self.selMu_etaCut = np.delete(self.selMu_eta, self.badEvents, axis=0)
        self.selMu_phiCut = np.delete(self.selMu_phi, self.badEvents, axis=0)
        self.selMu_ptCut = np.delete(self.selMu_pt, self.badEvents, axis=0)
        self.selMu_chargeCut = np.delete(self.selMu_charge, self.badEvents, axis=0)

        if self.dataset == "mc" or self.ml_met:
            self.genAMu_etaCut = np.delete(self.genAMu_eta, self.badEvents, axis=0)
            self.genAMu_phiCut = np.delete(self.genAMu_phi, self.badEvents, axis=0)
            self.genAMu_ptCut = np.delete(self.genAMu_pt, self.badEvents, axis=0)
            self.genAMu_chargeCut = np.delete(
                self.genAMu_charge, self.badEvents, axis=0
            )

        if ret:
            if self.dataset == "mc" or self.ml_met:
                if verbose:
                    print_alert(
                        "Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut, genAMu_etaCut, genAMu_phiCut, genAMu_ptCut, genAMu_chargeCut"
                    )

                return (
                    self.selMu_etaCut,
                    self.selMu_phiCut,
                    self.selMu_ptCut,
                    self.selMu_chargeCut,
                    self.genAMu_etaCut,
                    self.genAMu_phiCut,
                    self.genAMu_ptCut,
                    self.genAMu_chargeCut,
                )
            elif (
                self.dataset == "bkg"
                or self.dataset == "sig"
                or self.dataset == "ntuple"
            ):
                if verbose:
                    print_alert(
                        "Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut"
                    )

                return (
                    self.selMu_etaCut,
                    self.selMu_phiCut,
                    self.selMu_ptCut,
                    self.selMu_chargeCut,
                )

    def matchBkgMu(self, ret=False, verbose=False):
        """
        Arbitrarily "pseudo-matches" the muons from the background dataset (or any other
        dataset). This is a preprocessing step for data that will be fed to the
        trained ML model
        2022/08/30, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Change log:
        S.D.B., 2022/08/30
            - Added the printing to std. out in color via the print_alert() function in file_utils
        """

        numEvents = self.selMu_chargeCut.shape[0]
        self.min_dRgenFinal = np.ndarray((numEvents, 4))

        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Matching' %s muons" % self.dataset)
            print_alert(60 * "*")
            print_alert("\n\n")
            print_alert("Pseudo-matching the four muons for background data")

        def matchBkgMu(self, ret=False, verbose=False):

            """
            Arbitrarily "pseudo-matches" the muons from the background dataset (or any other
            dataset). This is a preprocessing step for data that will be fed to the
            trained ML model
            2022/08/30, v. 1

            Optional arguments:
            ret:        (bool) Return extraced data in the form of arrays. Default = False.
            verbose:    (bool) Increase verbosity. Default = False.

            Change log:
            S.D.B., 2022/08/30
                - Added the printing to std. out in color via the print_alert() function in file_utils
            """

            numEvents = self.selMu_chargeCut.shape[0]
            self.min_dRgenFinal = np.ndarray((numEvents, 4))

            if verbose:
                print_alert("\n\n")
                print_alert(60 * "*")
                print_alert("Matching' %s muons" % self.dataset)
                print_alert(60 * "*")
                print_alert("\n\n")
                print_alert("Pseudo-matching the four muons for background data")

            for event in range(numEvents):
                tempSelCharge = self.selMu_chargeCut[
                    event, :
                ]  # (1 x 4) extract charge data for all 4 muons and assign to temp array
                self.min_dRgenFinal[event, :] = np.array(
                    np.where(self.selMu_chargeCut[event, :])
                )  # 1 x 4 THIS DOESN'T LOOK RIGHT...INVESTIGATE FURTHER
                if (
                    tempSelCharge[0] == tempSelCharge[1]
                ):  # check if muons 0 and 1 have the same charge, if so, swap first with third muon
                    self.min_dRgenFinal[event, [0, 2]] = self.min_dRgenFinal[
                        event, [2, 0]
                    ]

            if (
                np.sum(self.min_dRgenFinal, axis=1).all()
                == np.full((numEvents), 6.0).all()
            ):  # check to make sure "matching" performed properly
                pass
            else:  # perform the 'matching' process again to fill array
                for event in range(numEvents):
                    tempSelCharge = self.selMu_chargeCut[event, :]  # 1 x 4
                    self.min_dRgenFinal[event,] = np.array(
                        np.where(selMu_chargeCut[event, :])
                    )  # 1 x 4
                    if tempSelCharge[0] == tempSelCharge[1]:
                        self.min_dRgenFinal[event, [0, 2]] = self.min_dRgenFinal[
                            event, [2, 0]
                        ]

                if (
                    np.sum(self.min_dRgenFinal, axis=1).all()
                    == np.full((numEvents), 6.0).all()
                ):  # check again to make sure "matching" performed properly
                    print_error("Error matching the background dimuons!\nExiting...")
                    sys.exit()  # if this hasn't been achieved by the second round of 'matching', exit

            if ret:
                return self.min_dRgenFinal

    def dRgenCalc(self, ret=False, verbose=False):
        """
        Calculates the dR value between the generator level and reco level muons.
        To be used to determine if the muons are reconstructed properly.
        Stephen D. Butalla & Mehdi Rahmani
        2022/08/30, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        genAMu_eta: np.ndarray; reco level muon eta
        selMu_eta:  np.ndarray; generator level muon eta
        genAMu_phi: np.ndarray; reco level muon phi
        selMu_phi:  np.ndarray; generator level muon phi

        Change log:
        S.D.B., 2022/08/30
            - Added the printing to std. out in color via the print_alert() function in file_utils.
        """
        self.dRgen = np.ndarray((self.genAMu_etaCut.shape[0], 4, 4))

        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Calculating dR between gen level and reco level muons (MC)")
            print_alert(60 * "*")
            print_alert("\n\n")

        for ii in tqdm(range(self.genAMu_etaCut.shape[0])):  # for each event
            for jj in range(4):  # dR for each gen muon
                for ll in range(4):  # for each sel muon
                    self.dRgen[ii, jj, ll] = sqrt(
                        pow(self.genAMu_etaCut[ii, jj] - self.selMu_etaCut[ii, ll], 2)
                        + pow(self.genAMu_phiCut[ii, jj] - self.selMu_phiCut[ii, ll], 2)
                    )  # calculate delta R between each gen and sel level muons
        if ret:
            return self.dRgen

    def SSM(self, ret=False, verbose=False, extraInfo=False):
        """
        Stochastic sampling method (SSM) used to pseudorandomly sample
        the muons and then determinine the minimum delta R between the gen
        and sel level muons. Outline of procedure:

        for each event:
            1. Retrieve both the sel and gen level charge information of the
               muons.
            2. Pseudorandomly select a muon [0, 3].
            3. Based off of this muon, determine the oppositely charged muons
               and the other muon with same charge.
            4. Calculate the minimum dR (dR = sqrt{eta^2 + phi^2}; previously
               calculated) between the gen and sel muons for same and opposite
               charge muons.
                   a. If pseudorandomly chosen muon has minimum dR of the two same/opposite
                      charged muons:
                          Label that muon and remove the index from the total list of charge.
                   b. Pair the other sel muon with the other same charge gen muon.
            5. From the minima previously calculated, determine which opposite charged
               muon has the minimum dR.
                   a. Pair the remaining oppositely charged sel muon to the last muon available.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/30, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.
        extraInfo:  (bool) Calculate extra information, e.g., the difference in the gen and sel
                    level pT, eta, and phi.

        Outputs:

        Change log:
        S.D.B., 2022/08/30
            - Added the printing to std. out in color via the print_alert() function in file_utils.

        """
        self.min_dRgen = np.ndarray((self.genAMu_chargeCut.shape[0], 4))
        self.dRgenMatched = np.ndarray((self.genAMu_chargeCut.shape[0], 4))

        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert(
                "Using tochastic sampling method (SSM) to match gen and reco level muons"
            )
            print_alert(60 * "*")
            print_alert("\n\n")

        for event in tqdm(range(self.selMu_chargeCut.shape[0])):  # loop over events

            tempGenCharge = self.genAMu_chargeCut[
                event, :
            ]  # (1 x 4) Pulls out the generator level charge for all four muons for one event
            tempSelCharge = self.selMu_chargeCut[
                event, :
            ]  # (1 x 4) Pulls out the generator level charge for all four muons for one event

            # Match first randomly chosen muon
            index = randint(0, 3)  # Generate random int in [0,3]
            chargeTemp = tempGenCharge[index]  # Get charge corresponding to AXMuY

            genCharge = np.array(
                np.where(self.genAMu_chargeCut[event, :] == chargeTemp)
            ).reshape(
                (2,)
            )  # Select gen muons where charge (array of indices)
            selCharge = np.array(
                np.where(self.selMu_chargeCut[event, :] == chargeTemp)
            ).reshape(
                (2,)
            )  # Select sel muons where charge is == AXMuY (array of indices)

            genChargeopo = np.array(
                np.where(self.genAMu_chargeCut[event, :] != chargeTemp)
            ).reshape(
                (2,)
            )  # Select gen muons where charge (array of indices)
            selChargeopo = np.array(
                np.where(self.selMu_chargeCut[event, :] != chargeTemp)
            ).reshape(
                (2,)
            )  # Select sel muons where charge (array of indices)

            # if verbose:
            #    print_alert("Event: %i\n" % event)
            # print charge of each sel muon
            #    for pair in range(2):
            #        print_alert("sel-level charge, muon%d: %d" % (pair, selCharge[pair]))
            #    for pair in range(2):
            #        print_alert("sel Opposite sel-level chargeopo[%d]: %d", selChargeopo[pair])

            # print charge of each gen muon
            #    for pair in range(2):
            #        print_alert("gen charge[%d]: %d", genCharge[pair])
            #    for pair in range(2):
            #        print_alert("gen chargeopo[0]: ", genChargeopo[0])
            #    print_alert("gen chargeopo[1]: ", genChargeopo[1])

            # Calculating minimum dR for each same and opposite charge gen muons
            min_dR0_index1 = np.array(
                np.minimum(
                    self.dRgen[event, genCharge[0], selCharge[0]],
                    self.dRgen[event, genCharge[0], selCharge[1]],
                )
            ).reshape((1,))
            min_dR0_index2 = np.array(
                np.minimum(
                    self.dRgen[event, genCharge[1], selCharge[0]],
                    self.dRgen[event, genCharge[1], selCharge[1]],
                )
            ).reshape((1,))
            min_dR0_index3 = np.array(
                np.minimum(
                    self.dRgen[event, genChargeopo[0], selChargeopo[0]],
                    self.dRgen[event, genChargeopo[0], selChargeopo[1]],
                )
            ).reshape((1,))
            min_dR0_index4 = np.array(
                np.minimum(
                    self.dRgen[event, genChargeopo[1], selChargeopo[0]],
                    self.dRgen[event, genChargeopo[1], selChargeopo[1]],
                )
            ).reshape((1,))

            # Calculating for the first gen muon
            if self.dRgen[event, genCharge[0], selCharge[0]] == min_dR0_index1:
                selIndex1 = selCharge[0]
                genIndex1 = genCharge[0]
            else:
                selIndex1 = selCharge[1]
                genIndex1 = genCharge[0]

            self.dRgenMatched[event, 0] = self.dRgen[event, genIndex1, selIndex1]

            temp = np.delete(selCharge, np.where(selCharge == selIndex1))

            genIndex2 = genCharge[1]
            selIndex2 = temp[0]

            self.dRgenMatched[event, 1] = self.dRgen[event, genIndex2, selIndex2]

            if self.dRgen[event, genChargeopo[0], selChargeopo[0]] == min_dR0_index3:
                selIndex3 = selChargeopo[0]
                genIndex3 = genChargeopo[0]
            else:
                selIndex3 = selChargeopo[1]
                genIndex3 = genChargeopo[0]

            self.dRgenMatched[event, 2] = self.dRgen[event, genIndex3, selIndex3]

            tempopo = np.delete(selChargeopo, np.where(selChargeopo == selIndex3))

            genIndex4 = genChargeopo[1]
            selIndex4 = tempopo[0]

            self.dRgenMatched[event, 3] = self.dRgen[event, genIndex4, selIndex4]

            genInd = np.array((genIndex1, genIndex2, genIndex3, genIndex4))
            selInd = np.array((selIndex1, selIndex2, selIndex3, selIndex4))

            for muon in range(4):
                self.min_dRgen[event, genInd[muon]] = selInd[muon]

            # if verbose:
            #    print_alert("sel muon: ", selIndex1, ", mached with gen: ", genIndex1)
            #    print_alert("other sel muon: ", selIndex2, ", mached with other gen: ", genIndex2)
            #    print_alert("opposite charge sel muon: ", selIndex3, ", mached with opposite charge gen: ", genIndex3)
            #    print_alert("other opposite charge sel muon: ", selIndex4, ", mached with other opposite charge gen: ", genIndex4)

            if extraInfo:
                dEtaMatched = np.ndarray((self.dRgen.shape[0], 4))
                dPhiMatched = np.ndarray((self.dRgen.shape[0], 4))
                dPtMatched = np.ndarray((self.dRgen.shape[0], 4))

                for muon in range(4):
                    dEtaMatched[event, muon] = (
                        self.genAMu_etaCut[event, genInd[muon]]
                        - self.selMu_etaCut[event, selInd[muon]]
                    )
                    dPhiMatched[event, muon] = (
                        self.genAMu_phiCut[event, genInd[muon]]
                        - self.selMu_phiCut[event, selInd[muon]]
                    )
                    dPtMatched[event, muon] = (
                        self.genAMu_ptCut[event, genInd[muon]]
                        - self.selMu_ptCut[event, selInd[muon]]
                    )

        if ret:
            if not extraInfo:
                if verbose:
                    print_alert("Arrays returned: min_dRgen, dRgenMatched")

                return self.min_dRgen, self.dRgenMatched
            else:
                if verbose:
                    print_alert(
                        "Arrays returned: min_dRgen, dRgenMatched, dEtaMatched, dPhiMatched, dPtMatched"
                    )

                return (
                    self.min_dRgen,
                    self.dRgenMatched,
                    dEtaMatched,
                    dPhiMatched,
                    dPtMatched,
                )

    def dRcut(self, dRcut=dR_cut, ret=False, verbose=False):
        """
        Apply cuts based on dR. Remove events if two muons---gen and sel level---are too far apart in R.
        The dR cut value is specified as a global variable at the top of the file utilitiesPlusPlus_v3.py.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        dRcut:      (float) The maximum value of dR for which events are retained. All events with dR greater
                    are removed. The default is the global variable defined at the top of the file.
                    ret = False, verbose = False
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        The arrays of observables with the final cuts applied:
        self.selMu_etaFinal
        self.selMu_phiFinal
        selMu_ptFinal
        selMu_chargeFinal
        genAMu_etaFinal
        genAMu_phiFinal
        genAMu_ptFinal
        genAMu_chargeFinal
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert(
                "Cutting events that do not match the criterion: dR < %.2f" % dRcut
            )
            print_alert(60 * "*")
            print_alert("\n\n")

        self.dRcut_events = np.unique(
            np.where(self.dRgenMatched > dRcut)[0]
        )  # get (unique) indices where the condition is satisfied

        # apply cuts to all relevant features
        self.selMu_etaFinal = np.delete(self.selMu_etaCut, self.dRcut_events, axis=0)
        self.selMu_phiFinal = np.delete(self.selMu_phiCut, self.dRcut_events, axis=0)
        self.selMu_ptFinal = np.delete(self.selMu_ptCut, self.dRcut_events, axis=0)
        self.selMu_chargeFinal = np.delete(
            self.selMu_chargeCut, self.dRcut_events, axis=0
        )
        self.genAMu_etaFinal = np.delete(self.genAMu_etaCut, self.dRcut_events, axis=0)
        self.genAMu_phiFinal = np.delete(self.genAMu_phiCut, self.dRcut_events, axis=0)
        self.genAMu_ptFinal = np.delete(self.genAMu_ptCut, self.dRcut_events, axis=0)
        self.genAMu_chargeFinal = np.delete(
            self.genAMu_chargeCut, self.dRcut_events, axis=0
        )
        self.min_dRgenFinal = np.delete(self.min_dRgen, self.dRcut_events, axis=0)
        self.dRgenMatched = np.delete(self.dRgenMatched, self.dRcut_events, axis=0)

        # Clean up memory
        del self.selMu_etaCut
        del self.selMu_phiCut
        del self.selMu_ptCut
        del self.selMu_chargeCut
        del self.genAMu_etaCut
        del self.genAMu_phiCut
        del self.genAMu_ptCut
        del self.genAMu_chargeCut

        if ret:
            if verbose:
                print(
                    "Arrays returned: selMu_etaCut, selMu_phiCut, selMu_ptCut, selMu_chargeCut, genAMu_etaCut, genAMu_phiCut, genAMu_ptCut, genAMu_chargeCut"
                )

            return (
                self.selMu_etaFinal,
                self.selMu_phiFinal,
                self.selMu_ptFinal,
                self.selMu_chargeFinal,
                self.genAMu_etaFinal,
                self.genAMu_phiFinal,
                self.genAMu_ptFinal,
                self.genAMu_chargeFinal,
                self.min_dRgenFinal,
                self.dRgenMatched,
            )

    def permutations(self, ret=False, verbose=False):
        """
        Generate all of the permutations possible of the four muons.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        Arrays of the wrong permuations and all permutations for each event:
        wrongPerm
        allPerm
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Generating all permutations")
            print_alert(60 * "*")

        if self.dataset == "bkg" or self.dataset == "sig" or self.dataset == "ntuple":
            print("Renaming sel mu arrays")
            # Rename to keep arrays consistent (no dR cut for bkg data)
            self.selMu_etaFinal = self.selMu_etaCut
            self.selMu_phiFinal = self.selMu_phiCut
            self.selMu_ptFinal = self.selMu_ptCut
            self.selMu_chargeFinal = self.selMu_chargeCut

            try:  # Clean up memory
                del self.selMu_etaCut
                del self.selMu_phiCut
                del self.selMu_ptCut
                del self.selMu_chargeCut
            except NameError:  # if they don't exist, e.g., if processing 'sig' dataset and ml_met is true, then pass
                pass

        if verbose:
            print_alert("Calculating all permutations of the 4 muons\n\n")
            print_alert("Calculating the wrong permutations")

        numEvents = self.min_dRgenFinal.shape[
            0
        ]  # int for total number of events for use in loop ranges
        self.wrongPerm = np.ndarray(
            (numEvents, 2, self.min_dRgenFinal.shape[1])
        )  # stores the incorrect permutations
        self.allPerm = np.ndarray(
            (numEvents, self.wrongPerm.shape[1] + 1, 4)
        )  # stores correct and incorrect permutations for each event

        for event in tqdm(range(numEvents)):
            self.wrongPerm[event, 0] = self.min_dRgenFinal[event, :]  # (1 x 4)
            self.wrongPerm[event, 1] = self.min_dRgenFinal[event, :]  # (1 x 4)

            correctPerm = np.array(
                self.min_dRgenFinal[event, :], dtype=int
            )  # Array of indices of matched data
            correctSelChargeOrder = self.selMu_chargeFinal[
                event, correctPerm
            ]  # Correct order of charges (matched)
            pos = np.array(np.where(correctSelChargeOrder == 1)).reshape(
                (2,)
            )  # indices of positive sel muons

            # if verbose:
            #    print_alert("Event %d" % event)
            #    print_alert("Correct permutation: ", correctPerm)
            #    print_alert("Correct sel muon charge order", correctSelChargeOrder)

            # start with permuting positive muons
            self.wrongPerm[event, 0, pos[0]] = self.min_dRgenFinal[event, pos[1]]
            self.wrongPerm[event, 0, pos[1]] = self.min_dRgenFinal[event, pos[0]]

            neg = np.array(np.where(correctSelChargeOrder == -1)).reshape((2,))

            # permute negative muons
            self.wrongPerm[event, 1, neg[0]] = self.min_dRgenFinal[event, neg[1]]
            self.wrongPerm[event, 1, neg[1]] = self.min_dRgenFinal[event, neg[0]]

        for event in tqdm(
            range(numEvents)
        ):  # store correct and incorrect permutations in an array with all permutations
            self.allPerm[event, 0, :] = self.min_dRgenFinal[
                event, :
            ]  # correct permutation
            self.allPerm[event, 1, :] = self.wrongPerm[
                event, 0, :
            ]  # wrong permutation 0
            self.allPerm[event, 2, :] = self.wrongPerm[
                event, 1, :
            ]  # wrong permutation 1

        self.allPerm = self.allPerm.astype(int)  # set all muon numbers to ints

        if ret:
            if verbose:
                print_alert("Arrays returned: wrongPerm, allPerm")

            return self.wrongPerm, self.allPerm

    def invMassCalc(self, ret=False, verbose=False):
        """
        Based on the correct pairing, calculate the invariant mass for the correct and
        incorrect permutations.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        Arrays of the wrong permuations and all permutations for each event:
        invariantMass: (attr; float, np.array) The invariant mass value for each dimuon pair for correct
                       and incorrect permutations, and the corresponding label (1 = correct, 0 = incorrect).
                       Shape: (N_events x 3 x 3). For one event, the contents are:
                       [invariantMass_0, invariantMass_1, label].
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Calculating the invariant mass for all permutations")
            print_alert(60 * "*")

        num = self.min_dRgenFinal.shape[0]
        self.invariantMass = np.ndarray((num, 3, 3))
        # invariantMass = np.ndarray((selMupTFinal.shape[0], permutationLabels.shape[0], permutationLabels.shape[1])

        for event in tqdm(range(num)):  # Loop over events
            ## pull out the indices of the paired muons and then stack to make a 1 x 4 np.array
            # Correct permutation indices
            A0_c = np.copy(
                self.min_dRgenFinal[event, 0:2]
            )  # (1 x 2 row vector) for the first correct pair of muons (pair A0)
            A1_c = np.copy(
                self.min_dRgenFinal[event, 2:4]
            )  # (1 x 2 row vector) for the second correct pair of muons (pair A1)

            # Incorrect permutation indices
            A0_w0 = np.copy(
                self.wrongPerm[event, 0, 0:2]
            )  # (1 x 2 row vector) for the first wrong pair of muons (pair A0), first incorrect permutation
            A1_w0 = np.copy(
                self.wrongPerm[event, 0, 2:4]
            )  # (1 x 2 row vector) for the second wrong pair of muons (pair A1), first incorrect permutation

            A0_w1 = np.copy(
                self.wrongPerm[event, 1, 0:2]
            )  # (1 x 2 row vector) for the first wrong pair of muons (pair A0), second incorrect permutation
            A1_w1 = np.copy(
                self.wrongPerm[event, 1, 2:4]
            )  # (1 x 2 row vector) for the second wrong pair of muons (pair A1), second incorrect permutation

            indicesA_c = np.column_stack(
                (A0_c, A1_c)
            )  # stack correct indices to make 1 x 4 row vector
            indicesA_c = indicesA_c.astype(int)  # convert to int

            indicesA_w0 = np.column_stack(
                (A0_w0, A1_w0)
            )  # stack indices of first incorrect permutation to make 1 x 4 row vector
            indicesA_w0 = indicesA_w0.astype(int)

            indicesA_w1 = np.column_stack(
                (A0_w1, A1_w1)
            )  # stack indices of second incorrect permutation to make 1 x 4 row vector
            indicesA_w1 = indicesA_w1.astype(int)

            # labels for correct/wrong pair; 1 = correct, 0 = incorrect
            self.invariantMass[event, 0, 2] = 1  # Correct permutation
            self.invariantMass[event, 1, 2] = 0  # Incorrect permutation 0
            self.invariantMass[event, 2, 2] = 0  # Incorrect permutation 1

            ##### CORRECT PERMUTATION #####
            # Calculation for A0
            self.invariantMass[event, 0, 0] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_c[0, 0]]
                    * self.selMu_ptFinal[event, indicesA_c[1, 0]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_c[0, 0]]
                        - self.selMu_etaFinal[event, indicesA_c[1, 0]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_c[0, 0]]
                        - self.selMu_phiFinal[event, indicesA_c[1, 0]]
                    )
                )
            )
            # Calculation for A1
            self.invariantMass[event, 0, 1] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_c[0, 1]]
                    * self.selMu_ptFinal[event, indicesA_c[1, 1]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_c[0, 1]]
                        - self.selMu_etaFinal[event, indicesA_c[1, 1]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_c[0, 1]]
                        - self.selMu_phiFinal[event, indicesA_c[1, 1]]
                    )
                )
            )

            ##### WRONG PERMUTATIONS #####
            # Calculation for A0
            self.invariantMass[event, 1, 0] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_w0[0, 0]]
                    * self.selMu_ptFinal[event, indicesA_w0[1, 0]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_w0[0, 0]]
                        - self.selMu_etaFinal[event, indicesA_w0[1, 0]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_w0[0, 0]]
                        - self.selMu_phiFinal[event, indicesA_w0[1, 0]]
                    )
                )
            )
            # Calculation for A1
            self.invariantMass[event, 1, 1] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_w0[0, 1]]
                    * self.selMu_ptFinal[event, indicesA_w0[1, 1]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_w0[0, 1]]
                        - self.selMu_etaFinal[event, indicesA_w0[1, 1]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_w0[0, 1]]
                        - self.selMu_phiFinal[event, indicesA_w0[1, 1]]
                    )
                )
            )

            # Calculation for A0
            self.invariantMass[event, 2, 0] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_w1[0, 0]]
                    * self.selMu_ptFinal[event, indicesA_w1[1, 0]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_w1[0, 0]]
                        - self.selMu_etaFinal[event, indicesA_w1[1, 0]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_w1[0, 0]]
                        - self.selMu_phiFinal[event, indicesA_w1[1, 0]]
                    )
                )
            )
            # Calculation for A1
            self.invariantMass[event, 2, 1] = np.sqrt(
                (
                    2
                    * self.selMu_ptFinal[event, indicesA_w1[0, 1]]
                    * self.selMu_ptFinal[event, indicesA_w1[1, 1]]
                )
                * (
                    np.cosh(
                        self.selMu_etaFinal[event, indicesA_w1[0, 1]]
                        - self.selMu_etaFinal[event, indicesA_w1[1, 1]]
                    )
                    - np.cos(
                        self.selMu_phiFinal[event, indicesA_w1[0, 1]]
                        - self.selMu_phiFinal[event, indicesA_w1[1, 1]]
                    )
                )
            )

        if ret:
            return self.invariantMass

    def dR_diMu(self, ret=False, verbose=False):
        """
        Calculates the dR between both sets of paired dimuons, and also calculates the
        dPhi between both sets of paired dimuons.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = False.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        Arrays of the wrong permuations and all permutations for each event:
        diMu_dR: (attr; float, np.array) The difference in R (R = sqrt{eta^2 + phi^2}) between the paired
                 muons in each pair.
        dPhi:    (attr; float, np.array) The difference in phi coordinate between the paired
                 muons in each pair.
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Calculating dPhi and dR for all permutations")
            print_alert(60 * "*")

        numEvents = self.min_dRgenFinal.shape[0]
        self.diMu_dR = np.ndarray((numEvents, 3, 2))
        self.dPhi = np.ndarray((numEvents, 3, 2))

        for event in tqdm(range(numEvents)):
            ## pull out the indices of the paired muons and then stack to make a 1 x 4 np.array
            # correct dR A0
            A0_c = np.copy(
                self.min_dRgenFinal[event, 0:2]
            )  # pull indices of first pair (1 x 2 np.array)
            A1_c = np.copy(
                self.min_dRgenFinal[event, 2:4]
            )  # pull indices of second pair (1 x 2 np.array)
            indicesA_c = np.column_stack((A0_c, A1_c))  # stack to make 1 x 4 np.array
            indicesA_c = indicesA_c.astype(int)  # set to int because these are indices

            # incorrect permutation 0
            A0_w0 = np.copy(
                self.wrongPerm[event, 0, 0:2]
            )  # pull indices of first pair (1 x 2 np.array)
            A1_w0 = np.copy(
                self.wrongPerm[event, 0, 2:4]
            )  # pull indices of second pair (1 x 2 np.array)
            indicesA_w0 = np.column_stack(
                (A0_w0, A1_w0)
            )  # stack to make 1 x 4 np.array
            indicesA_w0 = indicesA_w0.astype(
                int
            )  # set to int because these are indices

            # incorrect permutation 1
            A0_w1 = np.copy(
                self.wrongPerm[event, 1, 0:2]
            )  # pull indices of first pair (1 x 2 np.array)
            A1_w1 = np.copy(
                self.wrongPerm[event, 1, 2:4]
            )  # pull indices of second pair (1 x 2 np.array)
            indicesA_w1 = np.column_stack(
                (A0_w1, A1_w1)
            )  # stack to make 1 x 4 np.array
            indicesA_w1 = indicesA_w1.astype(
                int
            )  # set to int because these are indices

            ## difference in phi between the first two and second two muons
            # correct dPhi
            self.dPhi[event, 0, 0] = (
                self.selMu_phiFinal[event, indicesA_c[0, 0]]
                - self.selMu_phiFinal[event, indicesA_c[1, 0]]
            )  # dPhi of first pair of muons for correct pair
            self.dPhi[event, 0, 1] = (
                self.selMu_phiFinal[event, indicesA_c[0, 1]]
                - self.selMu_phiFinal[event, indicesA_c[1, 1]]
            )  # dPhi of second pair of muons for correct pair

            # incorrect dPhi 0
            self.dPhi[event, 1, 0] = (
                self.selMu_phiFinal[event, indicesA_w0[0, 0]]
                - self.selMu_phiFinal[event, indicesA_w0[1, 0]]
            )  # dPhi of first pair of muons for first incorrect pair
            self.dPhi[event, 1, 1] = (
                self.selMu_phiFinal[event, indicesA_w0[0, 1]]
                - self.selMu_phiFinal[event, indicesA_w0[1, 1]]
            )  # dPhi of second pair of muons for first incorrect pair

            # incorrect dPhi 0
            self.dPhi[event, 2, 0] = (
                self.selMu_phiFinal[event, indicesA_w1[0, 0]]
                - self.selMu_phiFinal[event, indicesA_w1[1, 0]]
            )  # dPhi of first pair of muons for second incorrect pair
            self.dPhi[event, 2, 1] = (
                self.selMu_phiFinal[event, indicesA_w1[0, 1]]
                - self.selMu_phiFinal[event, indicesA_w1[1, 1]]
            )  # dPhi of second pair of muons for second incorrect pair

            ## correct dR A0
            self.diMu_dR[event, 0, 0] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_c[0, 0]]
                    - self.selMu_etaFinal[event, indicesA_c[1, 0]],
                    2,
                )
                + pow(self.dPhi[event, 0, 0], 2)
            )  # dR of first pair of muons for correct pair
            ## correct dR A1
            self.diMu_dR[event, 0, 1] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_c[0, 1]]
                    - self.selMu_etaFinal[event, indicesA_c[1, 1]],
                    2,
                )
                + pow(self.dPhi[event, 0, 1], 2)
            )  # dR of second pair of muons for correct pair

            ## wrong0 dR A0
            self.diMu_dR[event, 1, 0] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_w0[0, 0]]
                    - self.selMu_etaFinal[event, indicesA_w0[1, 0]],
                    2,
                )
                + pow(self.dPhi[event, 1, 0], 2)
            )  # dR of first pair of muons for first incorrect pair
            ## wrong0 dR A1
            self.diMu_dR[event, 1, 1] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_w0[0, 1]]
                    - self.selMu_etaFinal[event, indicesA_w0[1, 1]],
                    2,
                )
                + pow(self.dPhi[event, 1, 1], 2)
            )  # dR of second pair of muons for first incorrect pair

            ## wrong1 dR A0
            self.diMu_dR[event, 2, 0] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_w1[0, 0]]
                    - self.selMu_etaFinal[event, indicesA_w1[1, 0]],
                    2,
                )
                + pow(self.dPhi[event, 2, 0], 2)
            )  # dR of first pair of muons for second incorrect pair
            ## wrong1 dR A1
            self.diMu_dR[event, 2, 1] = sqrt(
                pow(
                    self.selMu_etaFinal[event, indicesA_w1[0, 1]]
                    - self.selMu_etaFinal[event, indicesA_w1[1, 1]],
                    2,
                )
                + pow(self.dPhi[event, 2, 1], 2)
            )  # dR of second pair of muons for second incorrect pair

        if ret:
            return self.diMu_dR, self.dPhi

    def fillFinalArray(
        self,
        perm=True,
        diMu_dRBool=True,
        save=False,
        pandas=False,
        ret=True,
        verbose=False,
    ):
        """
        Fills the final output array for use with ML models or further analysis.

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        perm:              (bool) Add all permutation data to the final array. Default = True.
        diMu_dRBool        (bool) Add the dPhi and dR information to the final array. Default = True.
        save:              (bool) Save the data as a .csv (converst to a pd dataframe first). Default = True.
        pandas:            (bool) Convert np.array of data into a pd dataframe. Default = False. (Must be
                           True in order to save data as a .csv.)
        ret:               (bool) Return extraced data in the form of arrays. Default = True.
        verbose:           (bool) Increase verbosity. Default = False.

        Outputs:
        Arrays of the wrong permuations and all permutations for each event:
        dataframe_shaped   (attr; float, N_events x 24 np.array)
        total_df           (float, N x 24 pd.DataFrame)
        """
        # check args
        if save and not pandas:
            print_error(
                "Cannot save the final array of data unless it is converted to a pandas dataframe! Exiting...\n\n"
            )
            sys.exit()

        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Filling the final array for MC data")
            print_alert(60 * "*")

        if verbose:
            if pandas:
                print_alert("Filling final array/dataframe for use with XGBoost\n")

        if diMu_dRBool == True:
            # dataframe = np.ndarray((self.selMu_ptFinal.shape[0], self.invariantMass.shape[1], 23))
            dataframe = np.ndarray(
                (self.min_dRgenFinal.shape[0], 3, 24)
            )  # 3199 * 3 * 22
        else:
            dataframe = np.ndarray(
                (self.selMu_ptFinal.shape[0], self.invariantMass.shape[1], 19)
            )
        # print(dataframe.shape)

        if (
            perm == False and diMu_dRBool == True
        ):  # exclude permutations (only correct pair), onclude dR and dPhi for each paired muon
            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                for permutation in range(self.invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selMu_ptTemp = np.copy(self.selMu_ptFinal[event, :])  # 1 x 4
                    selMu_etaTemp = np.copy(self.selMu_etaFinal[event, :])  # 1 x 4
                    selMu_phiTemp = np.copy(self.selMu_phiFinal[event, :])  # 1 x 4
                    selMu_chargeTemp = np.copy(
                        self.selMu_chargeFinal[event, :]
                    )  # 1 x 4
                    invMass_temp = np.copy(
                        self.invariantMass[event, permutation, :]
                    )  # 1 x 3
                    dPhi_temp = np.copy(self.dPhi[event, permutation, :])
                    diMu_dR_temp = np.copy(self.diMu_dR[event, permutation, :])  # 1 x 2

                    dataframe[event, permutation, :] = np.concatenate(
                        (
                            selMu_ptTemp,
                            selMu_etaTemp,
                            selMu_phiTemp,
                            selMu_chargeTemp,
                            dPhi_temp,
                            diMu_dR_temp,
                            invMass_temp,
                        )
                    )

        if (
            perm == True and diMu_dRBool == False
        ):  # include all permutations, exclude dR and dPhi for each paired muon
            for event in tqdm(
                range(self.selMu_ptFinal.shape[0])
            ):  # loop over each event
                for permutation in range(
                    self.invariantMass.shape[1]
                ):  # loop over each of the permutations
                    selMu_ptTemp = np.copy(
                        self.selMu_ptFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_etaTemp = np.copy(
                        self.selMu_etaFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_phiTemp = np.copy(
                        self.selMu_phiFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_chargeTemp = np.copy(
                        self.selMu_chargeFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    invMass_temp = np.copy(
                        self.invariantMass[event, permutation, :]
                    )  # 1 x 3

                    dataframe[event, permutation, :] = np.concatenate(
                        (
                            selMu_ptTemp,
                            selMu_etaTemp,
                            selMu_phiTemp,
                            selMu_chargeTemp,
                            invMass_temp,
                        )
                    )

        if (
            perm == True and diMu_dRBool == True
        ):  # include all permutations, dR, and dPhi for each paired muon
            eventNum = np.ndarray((self.min_dRgenFinal.shape[0], 3))  # 3199 * 3 * 22
            counter = 0
            for event in range(self.min_dRgenFinal.shape[0]):
                print(counter)
                for ii in range(3):
                    eventNum[event, ii] = counter

                counter += 1

            eventNum = eventNum.astype(int)

            for event in tqdm(
                range(self.selMu_ptFinal.shape[0])
            ):  # loop over each event
                for permutation in range(
                    self.invariantMass.shape[1]
                ):  # loop over each of the permutations
                    eventNum_temp = np.copy(eventNum[event, permutation]).reshape((1,))
                    selMu_ptTemp = np.copy(
                        self.selMu_ptFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_etaTemp = np.copy(
                        self.selMu_etaFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_phiTemp = np.copy(
                        self.selMu_phiFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    selMu_chargeTemp = np.copy(
                        self.selMu_chargeFinal[event, self.allPerm[event, permutation]]
                    )  # 1 x 4
                    invMass_temp = np.copy(
                        self.invariantMass[event, permutation, :]
                    )  # 1 x 3
                    dPhi_temp = np.copy(self.dPhi[event, permutation, :])
                    diMu_dR_temp = np.copy(self.diMu_dR[event, permutation, :])  # 1 x 2

                    dataframe[event, permutation, :] = np.concatenate(
                        (
                            selMu_ptTemp,
                            selMu_etaTemp,
                            selMu_phiTemp,
                            selMu_chargeTemp,
                            dPhi_temp,
                            diMu_dR_temp,
                            eventNum_temp,
                            invMass_temp,
                        )
                    )

        if perm == False and diMu_dRBool == False:
            for event in tqdm(range(self.selMu_ptFinal.shape[0])):
                for permutation in range(self.invariantMass.shape[1]):
                    # Remove row of data for each sel muon
                    selMu_ptTemp = np.copy(self.selMu_ptFinal[event, :])  # 1 x 4
                    selMu_etaTemp = np.copy(self.selMu_etaFinal[event, :])  # 1 x 4
                    selMu_phiTemp = np.copy(self.selMu_phiFinal[event, :])  # 1 x 4
                    selMu_chargeTemp = np.copy(
                        self.selMu_chargeFinal[event, :]
                    )  # 1 x 4
                    invMass_temp = np.copy(
                        self.invariantMass[event, permutation, :]
                    )  # 1 x 3

                    dataframe[event, permutation, :] = np.concatenate(
                        (
                            selMu_ptTemp,
                            selMu_etaTemp,
                            selMu_phiTemp,
                            selMu_chargeTemp,
                            invMass_temp,
                        )
                    )  # 1 x 19

        self.dataframe_shaped = np.reshape(
            dataframe, (self.selMu_ptFinal.shape[0] * self.invariantMass.shape[1], 24)
        )

        if pandas:
            total_df = pd.DataFrame(
                self.dataframe_shaped,
                columns=[
                    "selpT0",
                    "selpT1",
                    "selpT2",
                    "selpT3",
                    "selEta0",
                    "selEta1",
                    "selEta2",
                    "selEta3",
                    "selPhi0",
                    "selPhi1",
                    "selPhi2",
                    "selPhi3",
                    "selCharge0",
                    "selCharge1",
                    "selCharge2",
                    "selCharge3",
                    "dPhi0",
                    "dPhi1",
                    "dRA0",
                    "dRA1",
                    "event",
                    "invMassA0",
                    "invMassA1",
                    "pair",
                ],
            )

            if save:
                dataDir = resultDir + "/" + self.fileName
                try:
                    os.makedirs(dataDir)  # create directory for VFAT data
                except FileExistsError:  # skip if directory already exists
                    pass

                total_df.to_csv(dataDir + "total_df_%s.csv" % self.fileName)

        if ret:
            if pandas:
                if verbose:
                    print_alertert("Arrays returned: total_df, dataframe_shaped")
                return total_df, self.dataframe_shaped
            else:
                if verbose:
                    print_alertert("Array returned: dataframe_shaped")
                return self.dataframe_shaped

    def fillAndSort(self, save=False, ret=True, verbose=False, custom_dir=None):
        """
        A lightweight version of fillFinalArray().

        Stephen D. Butalla & Mehdi Rahmani
        2022/08/31, v. 1

        Optional arguments:
        ret:        (bool) Return extraced data in the form of arrays. Default = True.
        verbose:    (bool) Increase verbosity. Default = False.

        Outputs:
        Final array with all information.
        dataframe_shaped: (attr; float/int, np.array) The array that has all of the event data. Shape is N_events x 24.

        Change log:
        S.D.B. 2022/08/31:
            - Added printing options from file_utils
            - Added custom directory option for saving dataframe as a .csv.
        """
        if verbose:
            print_alert("\n\n")
            print_alert(60 * "*")
            print_alert("Filling final array")
            print_alert(60 * "*")

        events = self.min_dRgenFinal.shape[0]
        eventNum = np.ndarray((events, 3))
        counter = 0

        for event in range(events):
            for ii in range(3):
                eventNum[event, ii] = counter
            counter += 1

        eventNum = eventNum.astype(int)

        # All data plus event numbers
        dataframe = np.ndarray((events, 3, 24))  # 3199 * 3 * 22

        for event in tqdm(range(self.selMu_ptFinal.shape[0])):
            for permutation in range(self.invariantMass.shape[1]):
                eventNum_temp = np.copy(eventNum[event, permutation]).reshape((1,))
                selpT_temp = np.copy(
                    self.selMu_ptFinal[event, self.allPerm[event, permutation]]
                )  # 1 x 4
                selEta_temp = np.copy(
                    self.selMu_etaFinal[event, self.allPerm[event, permutation]]
                )  # 1 x 4
                selPhi_temp = np.copy(
                    self.selMu_phiFinal[event, self.allPerm[event, permutation]]
                )  # 1 x 4
                selCharge_temp = np.copy(
                    self.selMu_chargeFinal[event, self.allPerm[event, permutation]]
                )  # 1 x 4
                invMass_temp = np.copy(
                    self.invariantMass[event, permutation, :]
                )  # 1 x 3
                dPhi_temp = np.copy(self.dPhi[event, permutation, :])
                diMu_dR_temp = np.copy(self.diMu_dR[event, permutation, :])  # 1 x 2

                dataframe[event, permutation, :] = np.concatenate(
                    (
                        selpT_temp,
                        selEta_temp,
                        selPhi_temp,
                        selCharge_temp,
                        dPhi_temp,
                        diMu_dR_temp,
                        eventNum_temp,
                        invMass_temp,
                    )
                )  # 1 x 19

        self.dataframe_shaped = np.reshape(
            dataframe, (self.selMu_ptFinal.shape[0] * self.invariantMass.shape[1], 24)
        )

        if self.dataset == "mc" or self.ml_met:
            if save:
                total_df = pd.DataFrame(
                    self.dataframe_shaped,
                    columns=[
                        "selpT0",
                        "selpT1",
                        "selpT2",
                        "selpT3",
                        "selEta0",
                        "selEta1",
                        "selEta2",
                        "selEta3",
                        "selPhi0",
                        "selPhi1",
                        "selPhi2",
                        "selPhi3",
                        "selCharge0",
                        "selCharge1",
                        "selCharge2",
                        "selCharge3",
                        "dPhi0",
                        "dPhi1",
                        "dRA0",
                        "dRA1",
                        "event",
                        "invMassA0",
                        "invMassA1",
                        "pair",
                    ],
                )

                if custom_dir is not None:
                    dataDir = custom_dir + "/" + self.fileName
                    try:
                        os.makedirs(dataDir)  # create directory
                    except FileExistsError:  # skip if directory already exists
                        pass
                else:
                    dataDir = resultDir + "/" + self.fileName
                    try:
                        os.makedirs(dataDir)  # create directory
                    except FileExistsError:  # skip if directory already exists
                        pass

                total_df.to_csv(dataDir + "/total_df_%s.csv" % self.fileName)

        if ret:
            return self.dataframe_shaped
