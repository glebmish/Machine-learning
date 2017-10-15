import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from matplotlib.colors import ListedColormap


def calc_f1(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(min(y_true.size, y_pred.size)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    return 2 * tp / (2 * tp + fp + fn)


def to_rad(dots):
    new_dots = []
    for dot in dots:
        new_dots.append([math.sqrt(dot[0] ** 2 + dot[1] ** 2), math.atan2(dot[1], dot[0])])
    return np.array(new_dots)


def to_tan(dots):
    new_dots = []
    for dot in dots:
        new_dots.append([math.tan(dot[0]), math.tan(dot[1])])
    return np.array(new_dots)


def minkowski_distance(x, y, p):
    ret = 0
    for i in range(0, min(x.size, y.size)):
        ret += (abs(x[i] - y[i])) ** p
    return ret ** (1 / p)


def kernel(x):
    sigma = 0.56
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x * x) / (2 * sigma ** 2))


def get_mean_by_neighbors(data_x, data_y, y_, k, p):
    dist_to_others = []
    for x_ in data_x:
        dist_to_others.append(minkowski_distance(y_, x_, p))
    ind_of_neighbors = np.argpartition(dist_to_others, k)[:k]
    max_dist = np.array(dist_to_others)[ind_of_neighbors].max()
    answer_0, answer_1 = 0, 0
    for ind, x_ in enumerate(data_x):
        x = dist_to_others[ind] / max_dist
        ker = kernel(x)
        if data_y[ind] == 0:
            answer_0 += ker
        else:
            answer_1 += ker
    return 1 if answer_1 > answer_0 else 0


def neighbors_classify(x_train, x_test, y_train, y_test, k, p):
    y_pred = []
    for x_test_e in x_test:
        y_pred.append(get_mean_by_neighbors(x_train, y_train, x_test_e, k, p))
    f1_measure = calc_f1(y_test, np.array(y_pred))
    print("k = " + str(k) + "       p = " + str(p) + "      f1_measure = " + str(f1_measure))
    return f1_measure


def generate_pcolormesh(xx, yy, k, p):
    mx, my = xx.flatten(), yy.flatten()
    classes = []
    for i in range(mx.size):
        x, y = mx[i], my[i]
        classes.append(get_mean_by_neighbors(X, y_.flatten(), np.array([x, y]), k, p))
    return classes


def show_(k, p):
    h = .02
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = generate_pcolormesh(xx, yy, 6, 4)
    Z = np.array(Z).reshape(xx.shape)

    cmap_light = ListedColormap(['#bacaf7', '#a3f89d'])
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # pluses = data[data[..., 2] == 1]
    pluses = X[y_ == 1]
    plt.plot(pluses[:, 0], pluses[:, 1], 'g+')
    minuses = X[y_ == 0]
    plt.plot(minuses[..., 0], minuses[:, 1], 'b_')
    plt.show()


data = np.loadtxt("chips.txt", delimiter=",")
X = data[..., [0, 1]]
y_ = data[..., 2]

# X = to_rad(X)
X = to_tan(X)

f1_max, k_max, p_max, folds_max = 0, 0, 0, 0
for folds in range(5, 12):
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for k in range(2, int(math.sqrt(data[..., 0].size))):
        for p in range(1, 3):
            measures = []
            for train_index, test_index in skf.split(data[..., [0, 1]], data[..., 2]):
                measures.append(neighbors_classify(X[train_index], X[test_index], y_[train_index], y_[test_index], k, p))
            if np.array(measures).mean() > f1_max:
                f1_max, k_max, p_max, folds_max = np.array(measures).mean(), k, p, folds

print(f1_max, k_max, p_max, folds_max)

# show_(k_max, p_max)
