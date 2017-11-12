from Bayes.document import Document
from Bayes.Bayes_classifier import Bayes
from Bayes.reader import read_buckets


if __name__ == "__main__":
    documents_buckets = read_buckets()

    train_set = sum(documents_buckets[:-2], [])
    test_set = documents_buckets[-1]

    bayes = Bayes()
    bayes.train(train_set)

    real_spam = 0
    predicted_spam = 0
    for doc in test_set:
        if doc.is_spam:
            real_spam += 1

        tpe = bayes.classify(doc)
        if tpe:
            predicted_spam += 1

        type_string = "spam" if tpe else "legit"
        print("{} classified as {}".format(doc.name, type_string))

    print("real: {}, predicted: {}".format(real_spam, predicted_spam))