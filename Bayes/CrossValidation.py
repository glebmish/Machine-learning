def validate(bayes, data, debug=False):
    # cross validation for 10 buckets

    f_measures = []

    for i in range(10):
        first_part = data[:9 - i]
        second_part = data[9 - i + 1:]
        train_set = first_part + second_part
        train_set = sum(train_set, [])
        test_set = data[9 - i]

        bayes.reset()
        bayes.train(train_set)

        f_measure = count_f(bayes, test_set, debug)
        f_measures.append(f_measure)

    return f_measures


def count_f(bayes, data, debug):
    # true positive is a spam which is predicted well
    # true negative is a not spam which is predicted well
    # false positive is a spam which is not predicted well
    # false negative is a not spam which is not predicted well
    # [[ true positive, true negative ],
    # [ false positive, false negative ]]
    counts = [[0, 0], [0, 0]]

    for doc in data:
        if doc.is_spam:
            if bayes.classify(doc):
                counts[0][0] += 1
                type_string = "spam"
            else:
                counts[1][0] += 1
                type_string = "legit"
        else:
            if bayes.classify(doc):
                counts[1][1] += 1
                type_string = "spam"
            else:
                counts[0][1] += 1
                type_string = "legit"

        if debug: print("{} classified as {}".format(doc.name, type_string))

    if counts[0][0] != 0:
        recall = counts[0][0] / (counts[0][0] + counts[1][1])
        precision = counts[0][0] / (counts[0][0] + counts[1][0])
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0
    return f_measure
