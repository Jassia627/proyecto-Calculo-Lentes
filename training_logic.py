import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

class NeuralNetworkTrainer:
    def __init__(self):
        self.net = None

    def initialize_model(self, hidden_layer_sizes, learning_rate, max_error):
        # Crear y configurar el modelo de red neuronal
        self.net = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1, warm_start=True,
                                learning_rate_init=learning_rate, tol=max_error, n_iter_no_change=10)

    def train_model(self, X_train, y_train, epochs):
        # Establecer el número de épocas y entrenar el modelo
        self.net.max_iter = epochs
        self.net.fit(X_train, y_train)

    def improve_training(self, X_train, y_train, additional_epochs):
        # Mejorar el entrenamiento del modelo añadiendo más épocas
        self.net.max_iter += additional_epochs
        self.net.fit(X_train, y_train)

    def get_weights_and_biases(self):
        # Obtener pesos y umbrales de la red neuronal
        weights = self.net.coefs_
        biases = self.net.intercepts_
        return weights, biases

    def predict(self, sample):
        # Realizar predicciones utilizando el modelo entrenado
        return self.net.predict(sample)[0]

    def get_scores(self, X_train, y_train, X_val, y_val):
        # Obtener puntajes de entrenamiento y validación
        train_score = self.net.score(X_train, y_train)
        val_score = self.net.score(X_val, y_val)
        return train_score, val_score

    def get_loss_curve(self):
        # Obtener la curva de pérdida del entrenamiento
        return self.net.loss_curve_

    def save_model(self, file_path):
        # Guardar el modelo entrenado
        joblib.dump(self.net, file_path)

    def load_model(self, file_path):
        # Cargar el modelo entrenado
        self.net = joblib.load(file_path)
