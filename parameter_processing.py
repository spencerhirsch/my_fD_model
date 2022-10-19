from xgboost import XGBClassifier


class Process:
    def select_file(self, eta, max_depth, resultDir, mc_model, booster):
        if booster is not None:
            dataDir = resultDir + "/" + mc_model + "/booster/booster=%s" % booster
        else:
            if eta is not None and max_depth is None:
                dataDir = (
                        resultDir + "/" + mc_model + "/eta/" + str(eta)[:3] + "/" + str(eta)
                )
            elif eta is None and max_depth is not None:
                dataDir = resultDir + "/" + mc_model + "/max_depth/" + str(max_depth)
            elif eta is not None and max_depth is not None:
                dataDir = resultDir + "/" + mc_model + ("/eta_and_max_depth/eta_%s/max_depth_%s/" % (eta, max_depth))
            else:
                dataDir = resultDir + "/" + mc_model + "/default_parameters"

        return dataDir

    def select_model(self, eta, max_depth, booster):
        model = XGBClassifier(
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=7,
            eta=eta,
            max_depth=max_depth,
            booster=booster
        )

        return model
