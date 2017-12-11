from kNN.kNNClassifier import KNNClassifier
import shared.Visualizer as vis
import shared.Validator as vld
import shared.Reader as reader
import shared.Splitter as splitter

if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)

    X_train, y_train, X_test, y_test = splitter.split_train_test(X, y, 0.8)

    knn = KNNClassifier()
    knn.train(X_train, y_train)

    print(vld.f1_measure(X_test, y_test, knn))
    print(vld.confusion_matrix(X_test, y_test, knn))

    y_guessed = [knn.predict(x) for x in X_test]
    vis.plot(X, y, X_test, y_guessed, knn, 100)
