import SupportVector.SVM as SVM
import SupportVector.Visualizer as Visualizer
import SupportVector.reader as reader

if __name__ == "__main__":

    X, y = reader.read_training_set(shuf=True)
    split = round(len(X) * 0.9)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    svm = SVM.SVM()
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("{} out of {} predictions correct".format(correct, len(y_predict)))

    Visualizer.visualize(X_train[y_train==1], X_train[y_train==-1], svm)