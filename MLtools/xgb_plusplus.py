import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef, f1_score
from xgboost import XGBClassifier
import sys
import os
import json
from parameter_processing import Process
from model import Model


global resultDir
resultDir = "/Volumes/SA Hirsch/Florida Tech/research/dataframes/"
# resultDir = "/dataframes/MZD_200_55_pd_model"
# resultDir = "dataframes"
model_list = []

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

        return None

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
        booster=None,
    ):

        num_of_epochs = 100

        print("\n\n")
        print(60 * "*")
        print(colors.GREEN + "Building the XGBoost model and training" + colors.ENDC)
        print(60 * "*")

        proc = Process()
        global dataDir
        if self.dataset == "mc":
            mc_model = filename.split(".")[0]
            # Original
            # dataDir = resultDir + "/" + mc_model

            dataDir = proc.select_file(eta, max_depth, resultDir, mc_model, booster)

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

        # build the classifier
        # Original
        # model = XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=7)

        model = proc.select_model(eta, max_depth, booster)

        # fit the classifier to the training data, set fix number of epochs to iterate over
        model.fit(self.trainX, self.trainY)
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
            class_report['parameters'] = {'eta': eta, 'max_depth': max_depth, 'booster': booster}
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
            model_list.append(mod)
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

            # class_out = dataDir + "/object.json"
            # out_file = open(class_out, "w")
            # json_str = json.dumps(mod.get_model())
            # json.dump(json_str, out_file)

            epochs = len(results["validation_0"]["logloss"])
            x_axis = range(0, epochs)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.grid()
            ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
            ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
            ax.legend()
            plt.ylabel("Log Loss", loc="top")
            plt.xlabel("Epoch", loc="right")
            plt.title("XGBoost Log Loss")

            if saveFig:
                plt.savefig(
                    dataDir + ("/XGBLogLoss_default_plusplus_%s.pdf" % self.file_name),
                    bbox_inches="tight",
                )

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
            results = model.evals_result()

            epochs = len(results["validation_0"]["error"])
            x_axis = range(0, epochs)

            # plot classification error
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.grid()
            ax.plot(x_axis, results["validation_0"]["error"], label="Train")
            ax.plot(x_axis, results["validation_1"]["error"], label="Test")
            ax.legend()
            plt.ylabel("Classification Error", loc="top")
            plt.xlabel("Epoch", loc="right")
            plt.title("XGBoost Classification Error")

            if saveFig:
                plt.savefig(
                    dataDir
                    + ("/XGBClassError_default_plusplus_%s.pdf" % self.file_name),
                    bbox_inches="tight",
                )

            # plt.show()

        if ret:
            if single_pair:
                return self.single_correct_pair
        else:
            return None

    def predict(
        self,
        dataframe_shaped,
        filename="MZD_200_55_pd_model.sav",
        single_pair=False,
        ret=False,
        verbose=True,
    ):
        if self.dataset == "bkg" or "sig":
            global dataDir
            dataDir = resultDir + "/" + self.file_name
            try:
                os.makedirs(dataDir)  # create directory for data/plots
            except FileExistsError:  # skip if directory already exists
                pass
        if self.dataset == "mc":
            raise ValueError(
                "Can only predict dimuon pairs from background or signal datasets"
            )
            sys.exit()

        if verbose:
            print("\n\n")
            print(60 * "*")
            print(
                colors.GREEN
                + "Loading trained XGBoost model from %s" % filename
                + colors.ENDC
            )
            print(60 * "*")

        loaded_model = joblib.load(filename)

        x_data = dataframe_shaped[:, 0:23]
        y_data = dataframe_shaped[:, 20:24]

        x_df = pd.DataFrame(
            x_data,
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

        y_df = pd.DataFrame(y_data, columns=["event", "invmA0", "invmA1", "pair"])
        x_sorted = x_df.sort_values("event")

        predY = loaded_model.predict(x_sorted)
        arr = predY

        x_sorted["Predict"] = arr.tolist()

        if verbose:
            print("\n\n")
            print(60 * "*")
            print(
                colors.GREEN
                + "Determining correct and wrong pairs for the %s dataset"
                % self.file_name
                + colors.ENDC
            )
            print(60 * "*")

        self.correct_pair = x_sorted[x_sorted["Predict"] == 1]
        self.wrong_pair = x_sorted[x_sorted["Predict"] == 0]

        self.correct_pair.to_csv(dataDir + ("/correct_pair_%s.csv" % self.file_name))
        self.wrong_pair.to_csv(dataDir + ("/wrong_pair_%s.csv" % self.file_name))

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

        if ret:
            return self.single_correct_pair

    def plotMatch(self, save=True, verbose=False):

        if verbose:
            print("\n\n")
            print(60 * "*")
            print(
                colors.GREEN
                + "Plotting correctly and incorrectly matched muons"
                + colors.ENDC
            )
            print(60 * "*")

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax = ax.ravel()

        for pair in range(2):
            if pair == 0:
                self.correct_pair.plot(
                    x="invMassA0",
                    y="invMassA1",
                    kind="scatter",
                    color="darkred",
                    ax=ax[pair],
                    zorder=3,
                )
                ax[pair].set_title("Correct Pair")
                ax[pair].set_ylim(0, 80)
                ax[pair].set_xlim(0, 80)
            elif pair == 1:
                self.wrong_pair.plot(
                    x="invMassA0",
                    y="invMassA1",
                    kind="scatter",
                    color="darkred",
                    ax=ax[pair],
                    zorder=3,
                )
                ax[pair].set_title("Wrong Pair")
                ax[pair].set_ylim(0, 250)
                ax[pair].set_xlim(0, 250)

            ax[pair].grid(zorder=0)
            ax[pair].set_xlabel(r"$m_{\mu\mu_{1}}$[GeV]", loc="right")
            ax[pair].set_ylabel(r"$m_{\mu\mu_{2}}$[GeV]", loc="top")

        if save:
            fig.savefig(
                dataDir + ("/2DInvMass_dPhiCor_%s.pdf" % self.file_name),
                bbox_inches="tight",
            )

        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True
        )
        ax = ax.ravel()
        partial = "invMassA"

        for pair in range(4):
            if pair <= 1:
                if pair == 0:
                    key = partial + "0"
                    ax[pair].set_xlabel(r"$m_{\mu\mu_{1}}$[GeV]", loc="right")
                elif pair == 1:
                    key = partial + "1"
                    ax[pair].set_xlabel(r"$m_{\mu\mu_{2}}$[GeV]", loc="right")
                self.correct_pair[key].plot.hist(
                    bins=100,
                    alpha=0.9,
                    range=(0, 100),
                    color="darkred",
                    ax=ax[pair],
                    zorder=3,
                )
                ax[pair].set_title("Correct Pair")
            elif pair > 1:
                if pair == 2:
                    key = partial + "0"
                    ax[pair].set_xlabel(r"$m_{\mu\mu_{1}}$[GeV]", loc="right")
                elif pair == 3:
                    key = partial + "1"
                    ax[pair].set_xlabel(r"$m_{\mu\mu_{2}}$[GeV]", loc="right")

                self.wrong_pair[key].plot.hist(
                    bins=100,
                    alpha=0.9,
                    range=(0, 200),
                    color="darkred",
                    ax=ax[pair],
                    zorder=3,
                )
                ax[pair].set_title("Wrong Pair")

            ax[pair].grid(zorder=0)
            ax[pair].set_ylabel("Frequency", loc="top")

        if save:
            fig.savefig(
                dataDir + ("/1DInvMass_dPhiCor_%s.pdf" % self.file_name),
                bbox_inches="tight",
            )
