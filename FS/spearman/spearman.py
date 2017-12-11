import numpy as np

def get_ranked(x):
    indices = np.argsort(x)
    x = sorted(x)
    ranks = []
    count = 1
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            count += 1
        else:
            for index in range(i - count, i):
                ranks.append(i + (1 - count) / 2 - 1)
            count = 1

    # in the end check for last occurrence
    i = len(x) - 1
    if x[i] == x[i - 1]:
        for index in range(i + 1 - count, i + 1):
            ranks.append(i + (1 - count) / 2)
    else:
        ranks.append(len(x) - 1)

    return np.array(ranks), indices


def get_relations(ranks):
    relations = []
    relations.append([ranks[0], 1])
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1]:
            relations[len(relations) - 1][1] += 1
        else:
            relations.append([ranks[i], 1])

    return np.array(relations)


def get_delta(ranks_X, ranks_Y):
    sum_x = 0
    sum_y = 0
    for i in range(len(ranks_X)):
        sum_x += ranks_X[i, 1] * (ranks_X[i, 1] * ranks_X[i, 1] - 1)
    for i in range(len(ranks_Y)):
        sum_x += ranks_Y[i, 1] * (ranks_Y[i, 1] * ranks_Y[i, 1] - 1)
    return (1 / 2) * (sum_x + sum_y)


def get_coherence(ranked_X, indices_X, ranked_Y, indices_Y, delta):
    sum = 0
    n = len(indices_X)
    for i in range(n):
        sum += (ranked_X[indices_X[i]] - (n + 1) / 2) * (ranked_Y[indices_Y[i]] - (n + 1) / 2)
    return sum / (n * (n - 1) * (n + 1) - delta)


def spearman(train_X, train_Y, count_treshold):
    attribute_indices = []

    # firstly count for Y
    ranked_Y, indices_Y = get_ranked(train_Y)
    relations_Y = get_relations(ranked_Y)

    for i in range(train_X.shape[1]):
        ranked_X, indices_X = get_ranked(train_X[:, i])
        relations_X = get_relations(ranked_X)
        delta = get_delta(relations_X, relations_Y)
        p = get_coherence(ranked_X, indices_X, ranked_Y, indices_Y, delta)
        if np.abs(p) > count_treshold:
            attribute_indices.append(i)
            print("Attribute " + str(i) + ": " + str(p))

    print(str(len(attribute_indices)) + " attributes passed treshold")

    return attribute_indices