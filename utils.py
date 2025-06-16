"""Módulo de utilerías para la charla."""

import plotly.graph_objects as go
from tqdm import tqdm
import numpy as np


def plot_students(horas_estudio, calif_previas, etiquetas):
    """Función para desplegar un scatter plot de los estudiantes.

    Parameters
    ----------
    horas_estudio : np.array
        Vector con horas de estudio por estudiante.
    calif_previas : np.array
        Vector con promedio de calificaciones previas por estudiante.
    etiquetas : np.array
        Vector del índice de aprobación.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Un gráfico interactivo de plotly.
    """

    # Creamos la figura
    fig = go.Figure()

    # Agregamos el gráfico de dispersión
    fig.add_trace(
        go.Scatter(x=horas_estudio,
                   y=calif_previas,
                   mode="markers",
                   marker=dict(color=etiquetas,
                               colorscale="RdYlGn",
                               opacity=0.6)))

    # Establecemos los límites de los ejes
    x_min, x_max = horas_estudio.min() - 1, horas_estudio.max() + 1
    y_min, y_max = calif_previas.min() - 1, calif_previas.max() + 1

    # Configuramos el diseño del gráfico
    fig.update_layout(
        height=400,
        width=600,
        title="Gráfico de dispersión de los datos",
        xaxis=dict(title="Horas de estudio", range=[x_min, x_max]),
        yaxis=dict(title="Calificaciones previas", range=[y_min, y_max]),
    )

    # Retornamos el gráfico
    return fig


def sigmoid(z):
    """Función de activación sigmoide.

    Parameters
    ----------
    z : np.array
        Vector de entrada.

    Returns
    -------
    np.float
        Output de función sigmoide.
    """

    return 1 / (1 + np.exp(-z))


def loss(y, y_pred):
    """Función de error cross-entropy.

    Parameters
    ----------
    y : np.array
        Vector de etiquetas reales.
    y : np.array
        Vector de etiquetas inferidas.

    Returns
    -------
    np.float
        Función de pérdida log-loss.
    """

    epsilon = 1e-8  # para evitar log(0)
    return -np.mean(y * np.log(y_pred + epsilon) +
                    (1 - y) * np.log(1 - y_pred + epsilon))


def train_neuron(X, y, lr=0.1, epochs=10000):
    """Función para crear y entrenar una neurona.

    Parameters
    ----------
    X : np.array
        Vector/matriz de características.
    y : np.array
        Vector de etiquetas reales.
    lr : float
        Taza de aprendizaje.
    epochs : int
        Número de épocas de entrenamiento.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Un gráfico interactivo de plotly.
    """

    # Inicialización de pesos y bias aleatorios
    n = X.shape[0]
    w = np.random.random(X.shape[1])
    b = np.random.random()

    # Creamos array para almacenar el histrial de error
    losses = []

    # Almacenaremos los valores de pesos y bias en cada paso
    # Esto lo utilizaremos para generar una animación del ajuste
    w_adjust = w.copy()
    b_adjust = np.array([b])

    # Entrenamiento de la neurona artificial
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Realizamos la operación de la neurona y aplicamos la función
        # de activación para generar una salida entre 0 y 1
        z = X @ w + b
        y_pred = sigmoid(z)

        # Medimos qué tan buena es la predicción y calculamos el error
        # Guardamos el valor del error en el histórico de entrenamiento
        current_loss = loss(y, y_pred)
        losses.append(current_loss)

        # Calculamos y actualizamos los gradientes
        dw = np.dot(X.T, (y_pred - y)) / n
        db = np.sum(y_pred - y) / n
        w -= lr * dw
        b -= lr * db

        # Actualizamos los valores de pesoso y bias.
        if epoch % 100 == 0:
            w_adjust = np.vstack((w_adjust, w))
            b_adjust = np.append(b_adjust, b)
            pbar.set_postfix({"Epoch": epoch, "Loss": current_loss})

    return losses, (w, b)


def plot_history(history):
    """Función para desplegar la historia de entrenamiento.

    Parameters
    ----------
    history : np.array
        Vector con la historia de entrenamiento.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Un gráfico interactivo de plotly.
    """

    # Creamos la figura
    fig = go.Figure()

    # Agregamos la línea del error de entrenamiento
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(history)),
            y=history,
            mode="lines",
            line=dict(color="royalblue"),
            name="Error de predicción",
            hovertemplate="Época: %{x}<br>Error: %{y:.4f}<extra></extra>"))

    # Configuramos diseño del gráfico
    fig.update_layout(width=700,
                      height=300,
                      title="Historia de entrenamiento",
                      xaxis_title="Épocas de entrenamiento",
                      yaxis_title="Error de predicción",
                      hovermode="x")

    # Retornamos gráfico
    return fig


def plot_decision_boundary(horas_estudio, calif_previas, etiquetas, w, b):
    """Función para desplegar un scatter plot de los estudiantes con
    frontera de decisión.

    Parameters
    ----------
    horas_estudio : np.array
        Vector con horas de estudio por estudiante.
    calif_previas : np.array
        Vector con promedio de calificaciones previas por estudiante.
    etiquetas : np.array
        Vector del índice de aprobación.
    w : np.array
        Vector de pesos del modelo [w1, w2].
    b : float
        Término independiente (bias) del modelo.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Un gráfico interactivo de plotly con la frontera de decisión.
    """

    # Crear figura
    fig = go.Figure()

    # Gráfico de dispersión de los datos
    fig.add_trace(
        go.Scatter(x=horas_estudio,
                   y=calif_previas,
                   mode="markers",
                   marker=dict(color=etiquetas,
                               colorscale="RdYlGn",
                               opacity=0.6),
                   name="Estudiantes"))

    # Frontera de decisión: y = -(w1*x + b)/w2
    x_vals = np.linspace(horas_estudio.min() - 1, horas_estudio.max() + 1, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    fig.add_trace(
        go.Scatter(x=x_vals,
                   y=y_vals,
                   mode="lines",
                   line=dict(color="black", dash="dash"),
                   name="Frontera de decisión"))

    # Límites del gráfico
    x_min, x_max = horas_estudio.min() - 1, horas_estudio.max() + 1
    y_min, y_max = calif_previas.min() - 1, calif_previas.max() + 1

    # Configuración de diseño
    fig.update_layout(height=400,
                      width=600,
                      title="Clasificación de estudiantes: ¿Aprueba o no?",
                      xaxis=dict(title="Horas de estudio",
                                 range=[x_min, x_max]),
                      yaxis=dict(title="Calificaciones previas",
                                 range=[y_min, y_max]),
                      legend=dict(x=0.01, y=0.99))

    return fig


def predict_student(horas_estudio, calif_previas, w, b):
    """Función para desplegar el índice de aprobación de un estudiante.

    Parameters
    ----------
    horas_estudio : float
        Valor de horas de estudio.
    calif_previas : float
        Valor del promedio de calificaciones previas.
    w : np.array
        Vector de pesos del modelo [w1, w2].
    b : float
        Término independiente (bias) del modelo.
    """

    x = np.array([horas_estudio, calif_previas])
    z = x @ w + b
    aprobacion = sigmoid(z)

    print(f"[INFO] Probabilidad de aprobación: {aprobacion:.4f}")
