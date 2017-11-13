from collections import defaultdict
from math import log, e

class Bayes(object):

    def __init__(self, min_occurence=5, discard_deviation=0.1,
                 include_subject=False, top_n=None):
        self.SPAM = 'spam'
        self.HAM = 'ham'

        self.min_occurence = min_occurence
        self.discard_deviation = discard_deviation
        self.include_subject = include_subject
        self.top_n = top_n

        self.data = {}
        self.all_ct = {}

        self.reset()

    def reset(self):
        self.data = defaultdict(lambda: defaultdict(lambda: 0.000001))
        self.all_ct = {self.SPAM: 0, self.HAM: 0}

    def train(self, train_set):
        for document in train_set:

            words = document.content
            if self.include_subject:
                words += document.subject

            self.adjust_coefs(words, document.is_spam)

    def adjust_coefs(self, words, is_spam):
        for word in words:
            if is_spam:
                self.data[word][self.SPAM] += 1
                self.all_ct[self.SPAM] += 1
            else:
                self.data[word][self.HAM] += 1
                self.all_ct[self.HAM] += 1

    def classify(self, document):
        def spam_prob(word):
            return self.data[word][self.SPAM] / (self.data[word][self.SPAM] + self.data[word][self.HAM])

        words = document.content
        if self.include_subject:
            words += document.subject

        words = list(filter(lambda word: word in self.data, words))

        if self.min_occurence is not None:
            words = list(filter(lambda word: self.data[word][self.SPAM] + self.data[word][self.HAM] >= self.min_occurence, words))

        if self.discard_deviation is not None:
            words = list(filter(lambda word: abs(0.5 - spam_prob(word)) >= self.discard_deviation, words))

        if self.top_n is not None:
            words.sort(key=lambda word: abs(0.5 - spam_prob(word)), reverse=True)
            words = words[:self.top_n]

        spam_probability = sum(map(lambda word: log(self.data[word][self.SPAM] / (self.all_ct[self.SPAM])), words))
        ham_probability = sum(map(lambda word: log(self.data[word][self.HAM] / (self.all_ct[self.HAM])), words))

        return spam_probability > ham_probability




