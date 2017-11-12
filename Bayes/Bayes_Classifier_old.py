# coding=utf-8
from collections import defaultdict
from math import log


class BayesClassifier(object):
    def __init__(self, feature_resolver):
        self.features = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.000001)))
        self.get_features = feature_resolver
        self.count = defaultdict(lambda: 0.0)

    def train(self, dataset):
        """Collects statistics from training dataset"""
        for value, type in dataset:
            features_list = self.get_features(value)
            for feature in features_list:
                self.features[type][feature][features_list[feature]] += 1
            self.count[type] += 1

        for type in self.features:
            for feature in self.features[type]:
                sum_feat = sum(self.features[type][feature].values())
                for letter in self.features[type][feature]:
                    self.features[type][feature][letter] /= sum_feat
        sum_ct = sum(self.count.values())
        for type in self.count:
            self.count[type] /= sum_ct

    def check(self, dataset):
        """Checks the false percent in testing dataset. Returns false percent and false records"""
        pos_neg_table = dict( (t, dict([(t, 0) for t in self.features])) for t in self.features)
        for value, type in dataset:
            cl_type = self.classify(value)
            pos_neg_table[type][cl_type] += 1
        recalls = {}
        precisions = {}
        for type in self.features:
            recalls[type] = float(pos_neg_table[type][type]) / (sum(pos_neg_table[type].values()) + 0.001)
            precisions[type] = float(pos_neg_table[type][type]) / \
                               (sum(pos_neg_table[cl_type][type] for cl_type in pos_neg_table) + 0.001)
        recall = sum(recalls.values()) / len(recalls)
        precision = sum(precisions.values()) / len(precisions)
        return recall, precision

    def classify(self, value):
        """Classifies value based on collected statistics"""
        value = value.lower()
        val_features = self.get_features(value)

        max_type = None
        max_prob = -1000

        for type in self.features:
            prob = log(self.count[type])

            for featName in self.features[type]:
                prob += log(self.features[type][featName][val_features[featName]])
            if max_prob < prob:
                max_prob = prob
                max_type = type

        return max_type
