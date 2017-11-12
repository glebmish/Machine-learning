from collections import defaultdict
from math import log, e

class Bayes(object):

    def __init__(self, spam_threshold=0.7, spam_probability=0.5,
                 min_occurence=5, discard_deviation=0.1, subject_importance=1,
                 top_n_subject=None, top_n_content=None):
        self.SPAM = 'spam'
        self.ALL = 'all'

        self.spam_threshold = spam_threshold
        self.spam_probability = spam_probability
        self.min_occurence = min_occurence
        self.discard_deviation = discard_deviation
        self.subject_importance = subject_importance
        self.top_n_subject = top_n_subject
        self.top_n_content = top_n_content

        self.subject_data = defaultdict(lambda: defaultdict(lambda: 0.000001))
        self.subject_words_ct = [0]
        self.content_data = defaultdict(lambda: defaultdict(lambda: 0.000001))
        self.content_words_ct = [0]

    def train(self, train_set):
        for document in train_set:
            self.adjust_coefs(self.subject_data, self.subject_words_ct, document.subject, document.is_spam)
            self.adjust_coefs(self.content_data, self.content_words_ct, document.content, document.is_spam)

        self.normalize_coefs(self.subject_data)
        self.normalize_coefs(self.content_data)

    def adjust_coefs(self, data, words_ct, words, is_spam):
        for word in words:
            if is_spam:
                data[word][self.SPAM] += 1
            data[word][self.ALL] += 1
            words_ct[0] += 1

    def normalize_coefs(self, data):
        for word in data:
            data[word][self.SPAM] /= data[word][self.ALL]

    def classify(self, document):
        subject_type = self.classify_part(self.subject_data, self.subject_words_ct, self.top_n_subject, document.subject)
        content_type = self.classify_part(self.content_data, self.content_words_ct, self.top_n_content, document.content)

        combined_type = (content_type + self.subject_importance * subject_type) / (self.subject_importance + 1)
        return combined_type > self.spam_threshold

    def classify_part(self, data, words_ct, top_n, words):
        words = list(filter(lambda word: word in data, words))

        if self.min_occurence is not None:
            words = list(filter(lambda word: data[word][self.ALL] >= self.min_occurence, words))

        if self.discard_deviation is not None:
            words = list(filter(lambda word: not(0.5 - self.discard_deviation <= data[word][self.SPAM] <= 0.5 + self.discard_deviation),
                                words))

        if top_n is not None:
            words.sort(key=(lambda word: data[word][self.ALL]), reverse=True)
            words = words[:top_n]

        spam_prob = log(self.spam_probability * words_ct[0])
        for word in words:
            spam_prob += log(data[word][self.SPAM] / data[word][self.ALL])

        return e ** spam_prob




