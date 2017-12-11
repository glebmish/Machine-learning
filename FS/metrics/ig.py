import numpy as np
from FS.metrics.base import Base


class IG(Base):

    def __init__(self, treshold):
        super().__init__(treshold)

    @staticmethod
    def ig(X, Y):
        S = 0
        H_c = 0
        Sum = 0
        M = 0

        x_dict, x_dict_pos, x_dict_neg, y_dict = IG.init_dicts(X, Y)

        for x in x_dict.keys():
            p = x_dict[x] / len(X)
            H_c = S + IG.entropy(p)
            S = H_c

        for y in y_dict.keys():
            p = y_dict[y] / len(Y)
            Sum = S + IG.entropy(p)
            S = Sum

        for x in x_dict.keys():
            for y in y_dict.keys():
                p = 0
                if y > 0:
                    p = (x_dict_pos[x] / sum(x_dict_pos.values())) / (y_dict[y] / len(Y))
                else:
                    p = (x_dict_neg[x] / sum(x_dict_neg.values())) / (y_dict[y] / len(Y))
                M = S + IG.entropy(p)
                S = M

        H = Sum * M
        return H_c - H


    @staticmethod
    def entropy(p):
        if p == 0: return 0
        return p * np.log2(p)


    @staticmethod
    def init_dicts(X, Y):
        x_dict_pos = {}
        x_dict_neg = {}
        x_dict = {}
        y_dict = {-1.0: 0, 1.0: 0}
        for i in range(len(X)):
            y_dict[Y[i]] += 1

            if X[i] in x_dict:
                x_dict[X[i]] += 1
            else:
                x_dict[X[i]] = 1

            if Y[i] > 0:
                if X[i] in x_dict_pos:
                    x_dict_pos[X[i]] += 1
                else:
                    x_dict_pos[X[i]] = 1
                if X[i] not in x_dict_neg:
                    x_dict_neg[X[i]] = 0

            else:
                if X[i] in x_dict_neg:
                    x_dict_neg[X[i]] += 1
                else:
                    x_dict_neg[X[i]] = 1
                if X[i] not in x_dict_pos:
                    x_dict_pos[X[i]] = 0

        return x_dict, x_dict_pos, x_dict_neg, y_dict


    def get_correlation_indices(self, train_X, train_Y):
        attribute_indices = []

        for i in range(train_X.shape[1]):
            p = IG.ig(train_X[:, i], train_Y)
            if np.abs(p) > self.treshold:
                attribute_indices.append(i)

        print(str(len(attribute_indices)) + " attributes passed treshold")

        return attribute_indices