

def split_train_test(X, y, train_fraction=0.9):
    split = round(train_fraction * len(X))

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test