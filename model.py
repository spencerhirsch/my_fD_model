class Model:
    def __init__(self):
        self.accuracy = None
        self.max_depth = None
        self.eta = None
        self.mcc = None

    def get_model(self):
        return {'Accuracy': self.accuracy, 'Max Depth': self.max_depth, 'Eta': self.eta, 'MCC': self.mcc}

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def set_eta(self, eta):
        self.eta = eta

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def set_mcc(self, mcc):
        self.mcc = mcc

    def get_accuracy(self):
        return self.accuracy

    def get_eta(self):
        return self.eta

    def get_max_depth(self):
        return self.max_depth

    def get_mcc(self):
        return self.mcc

