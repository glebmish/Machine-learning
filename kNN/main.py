from kNN.kNNClassifier import KNNClassifier
import shared.Visualizer as vis
import shared.Validator as vld
import shared.Reader as reader

if __name__ == "__main__":
    X, y = reader.read_training_set(shuf=True)
    knn = KNNClassifier()
    knn.train(X, y)

    print(vld.f1_measure(X, y, knn))
    print(vld.confusion_matrix(X, y, knn))

    vis.plot(X, y, knn, 50)
