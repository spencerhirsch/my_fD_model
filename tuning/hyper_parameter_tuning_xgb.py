import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, matthews_corrcoef
import sys
import os
import json

from sklearn.model_selection import train_test_split

from tuning.parameter_processing import Process
from tuning.model import Model
import time

'''
    Cleaned up version of xgb_plusplus.py for hyperparameter tuning. This version of the class is used
    strictly for tuning, all edits conflict wiht the original purpose of xg_plusplus.py. For organization 
    this new file was created so that it would be able to be added to the original repository.
    
    Given the processed data, the xgb function in the xgb class creates the model with the necessary
    parameters and dumps the history and classification report. As well as, create a new object to store
    a summary of the important information from the model. This includes the parameters as well as the
    necessary metrics in determining the efficiency of the model.
'''


resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/"


class colors:
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"
    ENDC = "\033[39m"


plt.rcParams.update({"font.size": 26})  # Increase font size


class xgb:
    model_list = []

    def __init__(self, dataset, file_name=None):
        if dataset not in ["mc", "bkg", "sig"]:
            print("Dataset type must be either 'mc' or 'bkg'.")
        else:
            self.dataset = dataset

        if self.dataset == "mc":
            self.file_name = "mc"
        elif self.dataset in ["bkg", "sig"]:
            if file_name is None:
                print(
                    colors.RED
                    + "The dataset name (e.g., file_name = qqToZZTo4L, file_name = DataAboveUpsilonCRSR_MZD_200_55_signal) must be provided!"
                )
                sys.exit()
            else:
                self.file_name = file_name

        try:
            os.makedirs(resultDir)  # create directory for results
        except FileExistsError:  # skip if directory already exists
            pass

    def split(self, dataframe_shaped, test=0.25, random=7, scalerType=None, ret=False):
        print("\n\n")
        print(60 * "*")
        print(colors.GREEN + "Splitting data into train/test datasets" + colors.ENDC)
        print(60 * "*")

        X_data = dataframe_shaped[:, 0:23]
        Y_data = dataframe_shaped[:, 20:24]

        if self.dataset == "mc":
            if scalerType is None:
                pass
            elif scalerType == "StandardScaler":
                scaler = StandardScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "MaxAbsScaler":
                scaler = MaxAbsScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "MaxAbsScaler":
                scaler = MinMaxScaler().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "Normalizer":
                scaler = Normalizer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "PowerTransformer":
                scaler = PowerTransformer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "QuantileTransformer":
                scaler = QuantileTransformer().fit(X_data)
                X_data = scaler.transform(X_data)
            elif scalerType == "RobustScaler":
                scaler = RobustScaler().fit(X_data)
                X_data = scaler.transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, Y_data, test_size=test, random_state=random
        )  # split into train/test datasets

        self.trainX = pd.DataFrame(
            X_train,
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
            ],
        )
        # self.trainX = self.trainX.drop(['event'], axis = 1)

        self.testX = pd.DataFrame(
            X_test,
            columns=["selpT0", "selpT1", "selpT2", "selpT3", "selEta0", "selEta1", "selEta2", "selEta3", "selPhi0",
                     "selPhi1", "selPhi2", "selPhi3", "selCharge0", "selCharge1", "selCharge2", "selCharge3", "dPhi0",
                     "dPhi1", "dRA0", "dRA1", "event", "invMassA0", "invMassA1"],)

        self.trainY = pd.DataFrame(
            y_train, columns=["event", "invmA0", "invmA1", "pair"]
        )
        self.trainY = self.trainY.drop(["event", "invmA0", "invmA1"], axis=1)
        self.testY = pd.DataFrame(y_test, columns=["event", "invmA0", "invmA1", "pair"])
        self.testY = self.testY.drop(["event", "invmA0", "invmA1"], axis=1)

        if ret:
            return self.trainX, self.testX, self.trainY, self.testY
        else:
            return None

    def xgb(
            self,
            met=True,
            save=True,
            filename="MZD_200_55_pd_model.sav",
            single_pair=False,
            ret=True,
            verbose=True,
            saveFig=True,
            eta=None,
            max_depth=None,
            booster='gbtree',
            reg_alpha=None,
            reg_lambda=None
    ):

        num_of_epochs = 100     # Still need to include the number of epochs to train the model on

        # default_dict = \
        #     {
        #         'eta': 0.3,
        #         'max_depth': 6,
        #         'booster': 'gbtree'
        #     }

        '''
        Don't really need this anymore because no None type value is passed for any of the parameters.
        '''

        # if eta is None:
        #     eta = default_dict['eta']
        #
        # if max_depth is None:
        #     max_depth = default_dict['max_depth']
        #
        # if booster is None:
        #     booster = default_dict['booster']

        print("\n\n")
        print(60 * "*")
        print(colors.GREEN + "Building the XGBoost model and training" + colors.ENDC)
        print(60 * "*")

        proc = Process()
        global dataDir
        if self.dataset == "mc":
            mc_model = filename.split(".")[0]
            dataDir = proc.select_file(eta, max_depth, resultDir, mc_model, reg_lambda, reg_alpha)
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "bkg":
            dataDir = resultDir + "/" + self.file_name
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        elif self.dataset == "sig":
            dataDir = resultDir + "/signal_MZD_"
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass

        model = proc.select_model(eta, max_depth, reg_lambda, reg_alpha)

        start = time.time()
        model.fit(self.trainX, self.trainY)
        end = time.time()

        total_time = end - start

        # model.fit(self.trainX, self.trainY, early_stopping_rounds=num_of_epochs)

        if save:
            # save the model to disk
            joblib.dump(model, filename)

        if met:
            predictedY = model.predict(self.testX)

            # Testing, original
            class_out = dataDir + "/classification_report.json"
            out_file = open(class_out, "w")
            class_report = dict(classification_report(self.testY, predictedY, output_dict=True))
            class_report['parameters'] = {'eta': eta, 'max_depth': max_depth, 'booster': booster,
                                          'l1': reg_alpha, 'l2': reg_lambda}
            class_report['mcc'] = matthews_corrcoef(self.testY, predictedY)
            json.dump(class_report, out_file)

            '''
                Filling the object that stores all of the data. Store the objects in a list to be sorted
                and outputted. Add to the model object for other parameters that will be tested.

                Booster?
                Check notebook for others.
            '''

            mod = Model()
            mod.set_eta(class_report['parameters']['eta'])
            mod.set_max_depth(class_report['parameters']['max_depth'])
            mod.set_booster(class_report['parameters']['booster'])
            mod.set_accuracy(class_report['accuracy'])
            mod.set_mcc(class_report['mcc'])
            mod.set_time(total_time)
            mod.set_f1(class_report['1.0']['f1-score'])
            mod.set_precision(class_report['1.0']['precision'])
            mod.set_reg_alpha(class_report['parameters']['l1'])
            mod.set_reg_lambda(class_report['parameters']['l2'])
            xgb.model_list.append(mod)
            del mod

            print(
                "\nTraining Classification Report:\n\n",
                classification_report(self.testY, predictedY),
            )

        # merging Xtrain and Xtest
        totalDF_X1 = pd.concat([self.trainX, self.testX], axis=0)

        # sorting the X based on event number
        self.sorted_X = totalDF_X1.sort_values("event")

        # predicting for ALL X
        total_predY = model.predict(self.sorted_X)

        # reshaping the
        pred = total_predY.reshape(-1, 3)

        # sorted_X
        arr = total_predY
        self.sorted_X["Predict"] = arr.tolist()

        # selecting rows based on condition
        # here we only have correct match for each event
        self.correct_pair = self.sorted_X[self.sorted_X["Predict"] == 1]

        # selecting the wrong matches for each event
        self.wrong_pair = self.sorted_X[self.sorted_X["Predict"] == 0]

        self.correct_pair.to_csv(dataDir + ("/correct_pair_%s.csv" % self.file_name))
        self.wrong_pair.to_csv(dataDir + ("/wrong_pair_%s.csv" % self.file_name))

        self.model_run = True

        if single_pair:
            self.single_correct_pair = self.correct_pair.drop_duplicates(
                subset=["event", "Predict"], keep="last"
            )
            self.single_wrong_pair = self.wrong_pair.drop_duplicates(
                subset=["event", "Predict"], keep="last"
            )

            self.single_correct_pair.to_csv(
                dataDir + ("/single_correct_pair_%s.csv" % self.file_name)
            )
            self.single_wrong_pair.to_csv(
                dataDir + ("/single_wrong_pair_%s.csv" % self.file_name)
            )

        if met:
            print("\n\n")
            print(60 * "*")
            print(
                colors.GREEN
                + "Retrieving metrics and plotting logloss and error"
                + colors.ENDC
            )
            print(60 * "*")

            # plot the logloss and error figures
            eval_set = [(self.trainX, self.trainY), (self.testX, self.testY)]
            model.fit(
                self.trainX,
                self.trainY,
                early_stopping_rounds=10,
                eval_metric="logloss",
                eval_set=eval_set,
                verbose=False,
            )

            class_out = dataDir + "/history.json"
            results = model.evals_result()
            out_file = open(class_out, "w")
            json.dump(results, out_file)

            # make predictions for test data
            y_pred = model.predict(self.testX)
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(self.testY, predictions)

            '''
                Probably don't need the following code. Grab the necessary metrics from the classification report.
                It is the more efficient and better way of doing it now that the classification report returns a
                dictionary object.
            '''

            print("Accuracy: %.2f%%" % (accuracy * 100.0))

            # epochs = len(results["validation_0"]["logloss"])
            # x_axis = range(0, epochs)

            # fig, ax = plt.subplots(figsize=(12, 12))
            # ax.grid()
            # ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
            # ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
            # ax.legend()
            # plt.ylabel("Log Loss", loc="top")
            # plt.xlabel("Epoch", loc="right")
            # plt.title("XGBoost Log Loss")
            #
            # if saveFig:
            #     plt.savefig(
            #         dataDir + ("/XGBLogLoss_default_plusplus_%s.pdf" % self.file_name),
            #         bbox_inches="tight",
            #     )

            # plt.show()
            # plt.clf()
            # plt.close()

            model.fit(
                self.trainX,
                self.trainY,
                early_stopping_rounds=10,
                eval_metric="error",
                eval_set=eval_set,
                verbose=False,
            )
            # results = model.evals_result()

            # epochs = len(results["validation_0"]["error"])
            # x_axis = range(0, epochs)

            # plot classification error
            # fig, ax = plt.subplots(figsize=(12, 12))
            # ax.grid()
            # ax.plot(x_axis, results["validation_0"]["error"], label="Train")
            # ax.plot(x_axis, results["validation_1"]["error"], label="Test")
            # ax.legend()
            # plt.ylabel("Classification Error", loc="top")
            # plt.xlabel("Epoch", loc="right")
            # plt.title("XGBoost Classification Error")
            #
            # if saveFig:
            #     plt.savefig(
            #         dataDir
            #         + ("/XGBClassError_default_plusplus_%s.pdf" % self.file_name),
            #         bbox_inches="tight",
            #     )

            # plt.show()

        if ret:
            if single_pair:
                return self.single_correct_pair
        else:
            return None
