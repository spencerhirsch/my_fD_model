class Model:
    def __init__(self):
        self.accuracy = None
        self.max_depth = None
        self.eta = None
        self.mcc = None
        self.booster = None
        self.time = None
        self.f1 = None
        self.precision = None

    def get_model(self):
        object_dict = \
            {
                'accuracy': self.accuracy,
                'max depth': self.max_depth,
                'eta': self.eta,
                'mcc': self.mcc,
                'booster': self.booster,
                'time': self.time,
                'f1': self.f1,
                'precision': self.precision
            }

        return object_dict

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def set_eta(self, eta):
        self.eta = eta

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def set_mcc(self, mcc):
        self.mcc = mcc

    def set_booster(self, booster):
        self.booster = booster

    def set_time(self, time):
        self.time = time

    def set_f1(self, f1):
        self.f1 = f1

    def set_precision(self, precision):
        self.precision = precision

    def get_accuracy(self):
        return self.accuracy

    def get_eta(self):
        return self.eta

    def get_max_depth(self):
        return self.max_depth

    def get_mcc(self):
        return self.mcc

    def get_booster(self):
        return self.get_booster()

    def get_time(self):
        return self.time

    def get_f1(self):
        return self.f1

    def get_precision(self):
        return self.precision