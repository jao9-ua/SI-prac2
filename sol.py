import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def show_image(imagen, title):
    plt.figure()
    plt.suptitle(title)
    plt.imshow(imagen, cmap='gray')
    plt.show()

def plot_X(X, title):
    plt.title(title)
    plt.plot(X)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.show()

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.rand()
        self.polarity = np.random.choice([-1, 1])

    def predict(self, X):
        predictions = np.ones(X.shape[0])
        if self.polarity == -1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions

class Adaboost:
    def __init__(self, T=5, A=20):
        self.T = T
        self.A = A
        self.classifiers = []

    def fit(self, X, Y, verbose=False):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for t in range(self.T):
            classifier = DecisionStump(n_features)
            predictions = classifier.predict(X)
            error = np.sum(weights * (predictions != Y))

            beta = 0.5 * np.log((1 - error) / error)
            weights *= np.exp(-beta * Y * predictions)
            weights /= np.sum(weights)

            self.classifiers.append((classifier, beta))

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for classifier, beta in self.classifiers:
            predictions += beta * classifier.predict(X)
        return np.sign(predictions)

# Código de prueba con datos MNIST
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Redimensiona las imágenes para aplanarlas y normalizar los píxeles
X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

# Convierte etiquetas a -1 y 1
Y_train_binary = np.where(Y_train == 6, 1, -1)
Y_test_binary = np.where(Y_test == 6, 1, -1)

# Entrenamiento y predicción con Adaboost
adaboost_classifier = Adaboost(T=20, A=10)
adaboost_classifier.fit(X_train_flat, Y_train_binary)

# Predicción en el conjunto de entrenamiento
predictions_train = adaboost_classifier.predict(X_train_flat)
accuracy_train = np.mean(predictions_train == Y_train_binary)

# Predicción en el conjunto de prueba
predictions_test = adaboost_classifier.predict(X_test_flat)
accuracy_test = np.mean(predictions_test == Y_test_binary)

# Imprime resultados
print(f"Tasa de acierto en el conjunto de entrenamiento: {accuracy_train * 100:.2f}%")
print(f"Tasa de acierto en el conjunto de prueba: {accuracy_test * 100:.2f}%")
