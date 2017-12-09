import pylab as pl


def visualize(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

    # w.x + b = 0
    a0 = -4;
    a1 = f(a0, clf.w, clf.b)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b)
    pl.plot([a0, b0], [a1, b1], "k")

    # w.x + b = 1
    a0 = -4;
    a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0, b0], [a1, b1], "k--")

    # w.x + b = -1
    a0 = -4;
    a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4;
    b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0, b0], [a1, b1], "k--")

    pl.axis("tight")
    pl.show()