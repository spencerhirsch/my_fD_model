from xgboost import XGBClassifier
import warnings
import os

class Process:

    '''
        Need to come up with a better file management system to store the data collected based on
        the hyperparamters, as more paramters are added the file_management is going to be crazy
        and hard to maintain.
    '''

    def select_file(self, eta, max_depth, resultDir, mc_model, reg_lambda, reg_alpha, objective):

        result_dir = resultDir + '/' + mc_model
        data_dir = result_dir + ('/eta_%s/max_depth_%s/l1_%s/l2_%s/objective_%s' % (eta, max_depth, reg_alpha, reg_lambda, objective))
        # data_dir = result_dir + ("/eta_%s_&_max_depth_%s_&_l1_%s_&_l2_%s/"
        #                                          % (eta, max_depth, reg_alpha, reg_lambda))



        return data_dir

    def select_model(self, eta, max_depth, reg_lambda, reg_alpha, objective) -> XGBClassifier:
        warnings.filterwarnings("ignore")
        model = XGBClassifier(
            n_jobs=-1,
            # use_label_encoder=False,
            eval_metric="logloss",
            random_state=7,
            eta=eta,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective=objective
        )

        return model
