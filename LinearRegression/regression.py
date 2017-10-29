
def stupid_fit(X, y):
    return (100, 200)


class LinRegression(object):

    def __init__(self, fit_method=stupid_fit):
        self.W = []
        self.__fit = fit_method

    def fit(self, train_set):
        X = [(x.area, x.rooms) for x in train_set]
        y = [x.price for x in train_set]

        self.W = self.__fit(X, y)

    def predict(self, flat):
        X = (flat.area, flat.rooms)
        y = sum([w * x for w, x in zip(self.W, X)])

        return y
