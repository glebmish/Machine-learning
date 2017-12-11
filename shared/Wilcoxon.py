from SupportVector.svm import *
from SupportVector.kernel import Kernel
from kNN.kNNClassifier import *
import shared.Reader as reader
import shared.Splitter as splitter


def wilcoxon(Y1, Y2):
    diffs = [y1 - y2 for y1, y2 in zip(Y1, Y2)]
    diffs = [y for y in diffs if y != 0]

    diffs_abs = [[y, abs(y)] for y in diffs]
    diffs_abs = sorted(diffs_abs, key=lambda x: x[1])

    ranks = range(1, len(diffs_abs) + 1)
    diffs_abs_ranks = [[da[0], da[1], rank] for da, rank in zip(diffs_abs, ranks)]
    diffs_abs_ranks = make_rels(diffs_abs_ranks)

    atypical = -sum(map(lambda dar: sign(dar[0]), diffs_abs_ranks))

    wc = 0
    for diff, _, rank in diffs_abs_ranks:
        if sign(atypical) == sign(diff):
            wc += rank

    return wc


def sign(a):
    if a == 0:
        return 0

    return 1 if a > 0 else -1


def make_rels(diffs_abs_ranks):
    begin = 0
    end = 0

    while end != len(diffs_abs_ranks) - 1:
        end += 1

        if diffs_abs_ranks[begin][1] == diffs_abs_ranks[end][1]:
            pass
        else:
            ranksum = sum(list(map(lambda x: x[2], diffs_abs_ranks))[begin:end])
            rank = ranksum / (end - begin)

            for i in range(begin, end):
                diffs_abs_ranks[i][2] = rank

            begin = end

    end += 1
    ranksum = sum(list(map(lambda x: x[2], diffs_abs_ranks))[begin:end])
    rank = ranksum / (end - begin)

    for i in range(begin, end):
        diffs_abs_ranks[i][2] = rank

    return diffs_abs_ranks


if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)
    X_train, y_train, X_test, y_test = splitter.split_train_test(X, y, 0.8)

    svm = SVM(Kernel.gaussian(0.12), 0.1)
    svm.train(X_train, y_train)
    svm_guessed = [svm.predict(np.array(sample).reshape(1, 2)) for sample in X_test]

    knn = KNNClassifier()
    knn.train(X_train, y_train)
    knn_guessed = [knn.predict(np.array(sample).reshape(1, 2)) for sample in X_test]

    wc = wilcoxon(svm_guessed, knn_guessed)
    print('wilcoxon=' + str(wc))
