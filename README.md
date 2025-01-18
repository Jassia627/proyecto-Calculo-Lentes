# Sistema Inteligente de Lentes de Contacto

¡Bienvenido al proyecto **Sistema Inteligente de Lentes de Contacto**! Este sistema utiliza el poder de las redes neuronales para analizar mediciones oculares y recomendar características ideales para lentes de contacto personalizados.

---

## 🚀 Características Principales

### 📂 Gestión de Datos
- **Carga de datos desde CSV**: Los datos se dividen automáticamente en conjuntos de entrenamiento, validación y prueba.
- **Visualización y análisis sencillo**.

### 🤖 Red Neuronal Personalizable
- Configuración flexible:
  - Número de capas ocultas y neuronas.
  - Funciones de activación (sigmoide, tangente hiperbólica, etc.).
  - Parámetros como épocas, tasa de aprendizaje y error máximo.
- Entrenamiento optimizado con scikit-learn (`MLPRegressor`).

### 🔮 Predicciones Personalizadas
- Basado en las mediciones:
  - **Curvatura de la córnea (mm).**
  - **Diámetro de la pupila (mm).**
  - **Prescripción óptica (dioptrías).**
- Resultados incluyen:
  - Grosor del lente.
  - Curvatura del lente.
  - Material recomendado con explicación detallada.

### 📈 Visualización y Resultados
- **Curva de pérdida** para monitorear el progreso del entrenamiento.
- **Diseño del modelo neuronal** representado gráficamente.
- Visualización de pesos y sesgos iniciales del modelo.

### 💾 Guardado y Carga de Modelos
- Guarda modelos entrenados en archivos.
- Carga modelos previamente entrenados para reutilización.

---

## 📋 Requisitos del Sistema

### 🔧 Tecnologías Utilizadas
- **Python 3.8+**
- **Bibliotecas Principales**:
  - `PyQt5`: Para la interfaz gráfica.
  - `scikit-learn`: Para la implementación de la red neuronal.
  - `matplotlib`: Para la visualización de datos.
  - `pandas`: Para la gestión de datos.
  - `numpy`: Para cálculos numéricos.

### 📦 Instalación de Dependencias
Asegúrate de tener `pip` instalado y ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

---

## 🚀 ¿Cómo Empezar?

### 1️⃣ Clona el Repositorio
```bash
git clone https://github.com/Jassia627/proyecto-Calculo-Lentes.git
cd proyecto-Calculo-Lentes
```

### 2️⃣ Ejecuta la Aplicación
```bash
python main.py
```

### 3️⃣ Explora la Interfaz
- **Archivo → Cargar Datos**: Sube un archivo CSV con tus mediciones oculares.
- **Entrenar Modelo**: Configura y entrena la red neuronal.
- **Hacer Cálculo**: Introduce mediciones específicas para obtener recomendaciones.


---

## 🤝 Creadores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Jassia627">
        <img src="https://github.com/Jassia627.png" width="100px;" alt="Juan Assia"/>
        <br />
        <sub><b>Juan Assia</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Josechaparro09">
        <img src="https://github.com/Josechaparro09.png" width="100px;" alt="Jose Chaparro"/>
        <br />
        <sub><b>Jose Chaparro</b></sub>
      </a>
    </td>
  </tr>
</table>

---

¡Gracias por usar **Sistema Inteligente de Lentes de Contacto**! Si tienes preguntas o sugerencias, no dudes en abrir un issue o enviar un mensaje. 😊
