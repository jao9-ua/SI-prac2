import numpy as np

class DecisionStump:
    def __init__(self, n_features):
        self.feature_index = np.random.randint(0, n_features)  # Selecciona una característica al azar
        self.threshold = np.random.rand()  # Selecciona un umbral al azar
        self.polarity = np.random.choice([-1, 1])  # Selecciona una polaridad al azar

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
        weights = np.ones(n_samples) / n_samples  # Iniciar pesos de las observaciones

        for t in range(self.T):
            classifier = DecisionStump(n_features)
            predictions = classifier.predict(X)
            error = np.sum(weights * (predictions != Y))

            # Actualizar pesos de las observaciones
            beta = 0.5 * np.log((1 - error) / error)
            weights *= np.exp(-beta * Y * predictions)
            weights /= np.sum(weights)

            self.classifiers.append((classifier, beta))

    def predict(self, X):
        # Implementa la predicción con el clasificador fuerte Adaboost
        predictions = np.zeros(X.shape[0])
        for classifier, beta in self.classifiers:
            predictions += beta * classifier.predict(X)
        return np.sign(predictions)