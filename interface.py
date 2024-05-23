import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFormLayout, QComboBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split  # <-- Asegúrate de importar esto
from training_logic import NeuralNetworkTrainer

class IntelligentLensSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        self.trainer = NeuralNetworkTrainer()
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Sistema Inteligente de Lentes de Contacto')
        self.setGeometry(100, 100, 1200, 800)

        # Menú de la barra superior
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Archivo')

        loadAct = QAction('Cargar Datos', self)
        loadAct.triggered.connect(self.load_data)
        fileMenu.addAction(loadAct)

        saveAct = QAction('Guardar Modelo', self)
        saveAct.triggered.connect(self.save_model)
        fileMenu.addAction(saveAct)

        loadModelAct = QAction('Cargar Modelo', self)
        loadModelAct.triggered.connect(self.load_model)
        fileMenu.addAction(loadModelAct)

        trainAct = QAction('Entrenar Modelo', self)
        trainAct.triggered.connect(self.show_train_panel)
        fileMenu.addAction(trainAct)

        improveAct = QAction('Mejorar Entrenamiento', self)
        improveAct.triggered.connect(self.improve_training)
        fileMenu.addAction(improveAct)

        predictAct = QAction('Hacer Cálculo', self)
        predictAct.triggered.connect(self.show_calc_panel)
        fileMenu.addAction(predictAct)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.create_train_panel()
        self.create_calc_panel()
        self.create_network_design_panel()
        self.create_weights_panel()

        # Estilos estéticos
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit {
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px;
            }
            QTextEdit {
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                font-weight: bold;
            }
            QComboBox {
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px;
            }
        """)

    def load_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccione un archivo CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if not file_path:
            return
        data = pd.read_csv(file_path)
        X = data[['curvatura_corneal', 'diametro', 'error_refractario']].values
        y = data[['grosor_lente', 'curvatura_lente', 'material_lente']].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

        QMessageBox.information(self, "Éxito", "Datos cargados y divididos correctamente.")

    def create_train_panel(self):
        self.train_panel = QWidget()
        layout = QGridLayout()

        self.hidden_layers = []
        self.hidden_layers_layout = QVBoxLayout()
        self.add_hidden_layer_btn = QPushButton('Agregar Capa Oculta')
        self.add_hidden_layer_btn.clicked.connect(self.add_hidden_layer)

        layout.addWidget(QLabel('Parámetros de Entrada'), 0, 0)
        layout.addWidget(QLabel('Nº Entradas: 3\nNº Salidas: 3\nNº Patrones: N/A'), 0, 1)
        layout.addWidget(QLabel('Configuración de la Red'), 1, 0)
        layout.addLayout(self.hidden_layers_layout, 1, 1)
        layout.addWidget(self.add_hidden_layer_btn, 2, 1)

        self.output_layer_size = QLineEdit()
        self.output_layer_activation = QComboBox()
        self.output_layer_activation.addItems(['F. Lineal', 'Sigmoide', 'Tangente Hiperbólica'])
        layout.addWidget(QLabel('Capa Salida'), 3, 0)
        layout.addWidget(self.output_layer_size, 3, 1)
        layout.addWidget(QLabel('Función de Activación'), 4, 0)
        layout.addWidget(self.output_layer_activation, 4, 1)

        self.num_epochs = QLineEdit()
        self.learning_rate = QLineEdit()
        self.max_error = QLineEdit()
        layout.addWidget(QLabel('Parámetros de Entrenamiento'), 5, 0)
        layout.addWidget(QLabel('Número de iteraciones'), 5, 1)
        layout.addWidget(self.num_epochs, 5, 2)
        layout.addWidget(QLabel('Rata de aprendizaje'), 6, 1)
        layout.addWidget(self.learning_rate, 6, 2)
        layout.addWidget(QLabel('Error máximo permitido'), 7, 1)
        layout.addWidget(self.max_error, 7, 2)

        self.train_button = QPushButton('Iniciar Entrenamiento')
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button, 8, 1, 1, 2)

        self.train_results = QTextEdit()
        self.train_results.setReadOnly(True)
        layout.addWidget(self.train_results, 9, 0, 1, 3)

        self.train_panel.setLayout(layout)
        self.layout.addWidget(self.train_panel)
        self.train_panel.hide()

    def create_calc_panel(self):
        self.calc_panel = QWidget()
        layout = QGridLayout()

        self.curvatura = QLineEdit()
        self.diametro = QLineEdit()
        self.error_refractario = QLineEdit()

        layout.addWidget(QLabel('Curvatura de la córnea (mm):'), 0, 0)
        layout.addWidget(self.curvatura, 0, 1)
        layout.addWidget(QLabel('Diámetro de la pupila (mm):'), 1, 0)
        layout.addWidget(self.diametro, 1, 1)
        layout.addWidget(QLabel('Prescripción óptica (dioptrías):'), 2, 0)
        layout.addWidget(self.error_refractario, 2, 1)

        self.predict_button = QPushButton('Predecir')
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button, 3, 0, 1, 2)

        self.calc_results = QTextEdit()
        self.calc_results.setReadOnly(True)
        layout.addWidget(self.calc_results, 4, 0, 1, 2)

        self.calc_panel.setLayout(layout)
        self.layout.addWidget(self.calc_panel)
        self.calc_panel.hide()

    def create_network_design_panel(self):
        self.network_design_panel = QWidget()
        self.network_design_layout = QVBoxLayout(self.network_design_panel)
        self.network_design_label = QLabel('Diseño del Modelo Neuronal')
        self.network_design_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.network_design_layout.addWidget(self.network_design_label)
        self.network_figure = plt.figure()
        self.network_canvas = FigureCanvas(self.network_figure)
        self.network_design_layout.addWidget(self.network_canvas)
        self.layout.addWidget(self.network_design_panel)
        self.network_design_panel.hide()

    def create_weights_panel(self):
        self.weights_panel = QWidget()
        self.weights_layout = QVBoxLayout(self.weights_panel)
        self.weights_label = QLabel('Inicialización de Pesos y Umbrales')
        self.weights_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.weights_layout.addWidget(self.weights_label)
        self.weights_text = QTextEdit()
        self.weights_text.setReadOnly(True)
        self.weights_layout.addWidget(self.weights_text)
        self.layout.addWidget(self.weights_panel)
        self.weights_panel.hide()

    def show_train_panel(self):
        self.calc_panel.hide()
        self.network_design_panel.show()
        self.weights_panel.show()
        self.train_panel.show()

    def show_calc_panel(self):
        self.train_panel.hide()
        self.network_design_panel.hide()
        self.weights_panel.hide()
        self.calc_panel.show()

    def add_hidden_layer(self):
        hidden_layer_layout = QHBoxLayout()
        num_neurons = QLineEdit()
        num_neurons.setPlaceholderText('Número de Neuronas')
        activation_function = QComboBox()
        activation_function.addItems(['Sigmoide', 'Tangente Hiperbólica', 'Gausiana'])
        hidden_layer_layout.addWidget(num_neurons)
        hidden_layer_layout.addWidget(activation_function)
        self.hidden_layers_layout.addLayout(hidden_layer_layout)
        self.hidden_layers.append((num_neurons, activation_function))

    def train_model(self):
        if not self.train_data or not self.val_data:
            QMessageBox.critical(self, "Error", "Por favor, cargue los datos antes de entrenar el modelo.")
            return

        try:
            # Recopilación de los parámetros de configuración
            hidden_layer_sizes = [int(layer[0].text()) for layer in self.hidden_layers if layer[0].text()]
            if not hidden_layer_sizes:
                raise ValueError('Debe agregar al menos una capa oculta.')
            
            epochs = int(self.num_epochs.text())
            learning_rate = float(self.learning_rate.text())
            max_error = float(self.max_error.text())

            X_train, y_train = self.train_data
            X_val, y_val = self.val_data

            # Inicializar y entrenar el modelo usando la lógica del entrenamiento
            self.trainer.initialize_model(hidden_layer_sizes, learning_rate, max_error)
            self.trainer.train_model(X_train, y_train, epochs)

            train_score, val_score = self.trainer.get_scores(X_train, y_train, X_val, y_val)

            # Mostrar los resultados del entrenamiento
            self.train_results.setPlainText(f'Modelo entrenado con éxito\n'
                                            f'Score de entrenamiento: {train_score:.2f}\n'
                                            f'Score de validación: {val_score:.2f}\n')

            self.plot_training_curve()
            self.plot_network_design(hidden_layer_sizes)
            self.display_initial_weights()
        except ValueError as e:
            QMessageBox.critical(self, "Error", f'Error en los datos de configuración: {e}')

    def improve_training(self):
        if self.trainer.net is None:
            QMessageBox.critical(self, "Error", "No hay un modelo entrenado para mejorar. Por favor, entrene el modelo primero.")
            return

        epochs = int(self.num_epochs.text())

        self.trainer.improve_training(self.train_data[0], self.train_data[1], epochs)

        QMessageBox.information(self, "Mejora de Entrenamiento", "Modelo mejorado y guardado con éxito")
        self.display_initial_weights()

    def plot_training_curve(self):
        plt.figure()
        plt.plot(self.trainer.get_loss_curve())
        plt.title('Curva de Pérdida')
        plt.xlabel('Iteraciones')
        plt.ylabel('Pérdida')
        plt.show()

    def plot_network_design(self, hidden_layer_sizes):
        self.network_figure.clear()
        ax = self.network_figure.add_subplot(111)
        layer_sizes = [3] + hidden_layer_sizes + [3]
        v_spacing = 1.0 / float(max(layer_sizes))
        h_spacing = 1.0 / float(len(layer_sizes) - 1)
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2
            layer_top_b = v_spacing * (layer_size_b - 1) / 2
            for m in range(layer_size_a):
                for k in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing, (n + 1) * h_spacing],
                                      [layer_top_a - m * v_spacing, layer_top_b - k * v_spacing], color='k')
                    ax.add_line(line)
                    ax.text((n + 0.5) * h_spacing, (layer_top_a - m * v_spacing + layer_top_b - k * v_spacing) / 2,
                            f'W{n+1}[{m},{k}]', fontsize=8, color='blue')
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2
            for m in range(layer_size):
                circle = plt.Circle((n * h_spacing, layer_top - m * v_spacing), v_spacing / 4, color='w', ec='k', zorder=4)
                ax.add_artist(circle)
                ax.text(n * h_spacing, layer_top - m * v_spacing, f'L{n+1}N{m+1}', ha='center', va='center', fontsize=8, color='black')
        ax.axis('off')
        self.network_canvas.draw()

    def display_initial_weights(self):
        weights, biases = self.trainer.get_weights_and_biases()
        weights_text = "Pesos y Umbrales Iniciales:\n\n"
        for i, layer in enumerate(weights):
            weights_text += f'W{i+1} [{layer.shape[0]}X{layer.shape[1]}] =\n{layer}\n\n'
        for i, intercept in enumerate(biases):
            weights_text += f'U{i+1} [{len(intercept)}] =\n{intercept}\n\n'
        self.weights_text.setPlainText(weights_text)

    def predict(self):
        if self.trainer.net is None:
            QMessageBox.critical(self, "Error", "No hay un modelo entrenado. Por favor, entrene el modelo primero.")
            return

        try:
            curvatura = float(self.curvatura.text())
            diametro = float(self.diametro.text())
            error_refractario = float(self.error_refractario.text())

            sample = np.array([[curvatura, diametro, error_refractario]])
            prediction = self.trainer.predict(sample)

            grosor_lente = prediction[0]
            curvatura_lente = prediction[1]
            material_lente = round(prediction[2])

            if material_lente == 1:
                recomendacion = 'lente de hidrogel blando'
                mensaje = 'Las lentes de hidrogel blando son muy cómodas y tienen una alta permeabilidad al oxígeno, adecuadas para uso diario.'
            elif material_lente == 2:
                recomendacion = 'lente de hidrogel de silicona'
                mensaje = 'Las lentes de hidrogel de silicona permiten una alta transmisión de oxígeno, ideales para usuarios que necesitan usarlas por períodos prolongados.'
            elif material_lente == 3:
                recomendacion = 'lente rígida permeable al gas (RGP)'
                mensaje = 'Las lentes RGP ofrecen una visión nítida y son adecuadas para usuarios con astigmatismo o queratocono.'
            elif material_lente == 4:
                recomendacion = 'lente esclerótica'
                mensaje = 'Las lentes escleróticas cubren una mayor parte del ojo y son ideales para condiciones oculares severas como el síndrome de ojo seco.'
            else:
                recomendacion = 'material desconocido'
                mensaje = 'El material predicho no coincide con los materiales conocidos. Por favor, consulte a un especialista.'

            self.calc_results.setPlainText(f'Grosor de la lente: {grosor_lente:.2f} mm\n'
                                           f'Curvatura de la lente: {curvatura_lente:.2f} mm\n'
                                           f'Material de la lente: {recomendacion}\n{mensaje}')
            self.display_initial_weights()
        except ValueError as e:
            QMessageBox.critical(self, "Error", f'Error en los datos de entrada: {e}')

    def save_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo", "", "Model Files (*.pkl);;All Files (*)", options=options)
        if not file_path:
            return
        self.trainer.save_model(file_path)
        QMessageBox.information(self, "Éxito", "Modelo guardado correctamente.")

    def load_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Modelo", "", "Model Files (*.pkl);;All Files (*)", options=options)
        if not file_path:
            return
        self.trainer.load_model(file_path)
        QMessageBox.information(self, "Éxito", "Modelo cargado correctamente.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = IntelligentLensSystem()
    mainWin.show()
    sys.exit(app.exec_())
