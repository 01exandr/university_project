import pywt
import time
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
# from scipy.stats import spearmanr
from pandastable import Table
from matplotlib import style
from tkinter import ttk
from math import sqrt
from joblib import Parallel, delayed

available_styles = sorted(style.available)


class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def correlation(self):  # Кореляція
        return np.corrcoef(self.y_true, self.y_pred[:, 0])[0, 1]

    def correlation_arr(self):  # Кореляція
        return np.corrcoef(self.y_true, self.y_pred)[0, 1]

    def r2(self):  # Коефіцієнт детермінації (R-squared, R^2) / Вказує, наскільки отримані спостереження підтверджують модель.
        return r2_score(self.y_true, self.y_pred)

    def mse(self):  # Mean Squared Error / Середня квадратична помилка
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):  # Root Mean Squared Error / Середньоквадратична помилка
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def mae(self):  # Mean Absolute Error / Середня абсолютна похибка
        return mean_absolute_error(self.y_true, self.y_pred)


# Клас для прогнозування часових рядів
class TimeSeriesPredictor:
    # Конструктор класу приймає вхідний сигнал, розмір вікна, співвідношення розділення та кількість епох
    def __init__(self, signal, window_size=10, split_ratio=0.7, epochs=20):
        # Зберігаємо вхідний сигнал, розмір вікна, співвідношення розділення та кількість епох
        self.signal = signal
        self.window_size = window_size
        self.split_ratio = split_ratio
        self.epochs = epochs
        tf.function(reduce_retracing=True)
        self.model = tf.keras.models.Sequential()  # Створюємо нейронну мережу
        self.model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=window_size))  # Додаємо скритий шар з 10 нейронами і активаційною функцією 'relu'
        self.model.add(tf.keras.layers.Dense(1))  # Додаємо вихідний шар
        self.model.compile(optimizer='adam', loss='mse')  # Компілюємо модель з використанням оптимізатора 'adam' і функції втрат 'mse'

    # Генеруємо вхідні та вихідні дані для моделі
    def generate_data(self):
        X = np.array([self.signal[i-self.window_size:i] for i in range(self.window_size, len(self.signal))])
        y = self.signal[self.window_size:]
        return X, y

    # Розділяємо дані на тренувальний та тестовий набори
    def split_data(self, X, y):
        split_idx = int(len(X) * self.split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test, split_idx

    # Навчаємо модель
    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, verbose=0)

    # Робимо прогноз за допомогою моделі
    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_path):  # зберігаємо модель у файл
        self.model.save(model_path)  # 'my_model.h5'

    def load_model(self, model_path):  # завантажуємо модель з файлу
        self.model = tf.keras.models.load_model(model_path)

    def improve_and_save(self, X_train, y_train, model_path):
        self.fit(X_train, y_train)  # тренуємо модель на нових даних
        self.save_model(model_path)  # зберігаємо покращену модель

    def load_and_predict(self, X, model_path):
        self.load_model(model_path)  # завантажуємо модель
        return self.predict(X)  # прогнозуємо нові значення

    # Візуалізуємо результати
    def plot(self, t, y_train, y_test, y_pred, split_idx, ES):
        plt.figure(figsize=(10, 6))
        plt.plot(t[self.window_size:split_idx + self.window_size], y_train,
                 label='Тренувальні дані')  # Виводимо тренувальні дані
        plt.plot(t[split_idx + self.window_size:], y_test, label='Тестові дані')  # Виводимо тестові дані
        plt.plot(t[split_idx + self.window_size:], y_pred, label='Прогнозовані дані')  # Виводимо прогнозовані дані
        metrics = Metrics(y_test, y_pred)
        f1, f2, f3, f4, f5 = metrics.correlation(), metrics.r2(), metrics.mse(), metrics.rmse(), metrics.mae()
        plt.text(0.01, 0.8, f'Correlation: {f1:.2f}\nR-squared: {f2:.2f}\nMSE: {f3:.2f}\nRMSE: {f4:.2f}\nMAE: {f5:.2f}'
                            f'\ntime: {ES:.3f}', transform=plt.gca().transAxes)
        plt.legend()  # Додаємо легенду до графіку
        plt.grid(True)  # Додаємо сітку
        plt.show()  # Показуємо графік


class ARIMAPredictor:
    def __init__(self, signal, arima_order=(5, 1, 0), split_ratio=0.7):
        self.signal = signal
        self.arima_order = arima_order
        self.split_ratio = split_ratio

        self.train_data, self.test_data = self.split_data()
        self.len_test_data = len(self.test_data)
        self.history = np.array([x for x in self.train_data])
        self.forecast = np.array([])

    def split_data(self):
        split_idx = int(len(self.signal) * self.split_ratio)
        train, test = self.signal[:split_idx], self.signal[split_idx:]
        return train, test

    def t_predict(self):
        for time_point in range(self.len_test_data):
            model = ARIMA(self.history, order=self.arima_order)
            model_fit = model.fit()
            output = model_fit.forecast()
            print(output)
            yhat = output[0]
            self.forecast = np.append(self.forecast, yhat)
            # true_test_value = self.test_data[time_point]
            self.history = np.append(self.history, yhat)

    def plot(self, train, test, ES):
        plt.figure(figsize=(10, 6))
        plt.plot(train, label='Training Data')
        plt.plot(np.arange(len(train), len(train) + len(test)), test, label='Test Data')
        plt.plot(np.arange(len(train), len(train) + len(test)), self.forecast, label='Predicted Data')
        metrics_obj = Metrics(test, self.forecast)
        f1, f2, f3, f4, f5 = metrics_obj.correlation_arr(), metrics_obj.r2(), metrics_obj.mse(), metrics_obj.rmse(), metrics_obj.mae()
        plt.text(0.01, 0.8, f'Correlation: {f1:.2f}\nR-squared: {f2:.2f}\nMSE: {f3:.2f}\nRMSE: {f4:.2f}\nMAE: {f5:.2f}'
                            f'\ntime: {ES:.3f}', transform=plt.gca().transAxes)
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        start = time.time()
        self.t_predict()
        end = time.time()
        self.plot(self.train_data, self.test_data, end-start)


# Визначаємо клас WaveletTransform
class WaveletTransform:
    # Конструктор приймає сигнал, вейвлет ('db1' за замовчуванням) та рівень розкладання (якщо не вказано, вибирається максимально можливий)
    def __init__(self, signal, wavelet='db1', level=None):
        self.signal = signal  # Зберігаємо сигнал
        self.wavelet = pywt.Wavelet(wavelet)  # Створюємо об'єкт Wavelet з вказаним ім'ям
        if level is None:  # Якщо рівень не вказано, обчислюємо максимальний можливий рівень розкладання
            level = pywt.dwt_max_level(data_len=len(signal), filter_len=self.wavelet.dec_len)
        self.coeffs = pywt.wavedec(self.signal, self.wavelet, level=level)  # Виконуємо вейвлет-розкладання сигналу з обраним рівнем розкладання

    # Функція для отримання апроксимації сигналу на вказаному рівні
    def get_approximation(self, level):
        if level <= len(self.coeffs) - 1:  # Перевіряємо, чи є розкладання на цьому рівні
            return pywt.waverec(self.coeffs[:-level] + [None] * level, self.wavelet)  # Якщо так, повертаємо апроксимацію сигналу на цьому рівні
        else:
            return None  # Якщо ні, повертаємо None

    # Функція для отримання коефіцієнтів деталізації на вказаному рівні
    def get_details(self, level):
        if level <= len(self.coeffs) - 1:  # Перевіряємо, чи є розкладання на цьому рівні
            return self.coeffs[-level]  # Якщо так, повертаємо коефіцієнти деталізації на цьому рівні
        else:
            return None  # Якщо ні, повертаємо None


class ScrollableFrame(tk.Frame):  # Визначаємо клас ScrollableFrame, який наслідується від tk.Frame
    def __init__(self, container, *args, **kwargs):  # Ініціалізація класу
        super().__init__(container, *args, **kwargs)  # Викликаємо конструктор батьківського класу
        canvas = tk.Canvas(self)  # Створюємо екземпляр Canvas, що дозволяє малювати графічні об'єкти
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)  # Створюємо вертикальний Scrollbar, який пов'язаний з canvas
        self.scrollable_frame = ttk.Frame(canvas)  # Створюємо Frame всередині Canvas, який буде прокручуватися

        # Прив'язуємо метод конфігурації canvas до події <Configure> на scrollable_frame.
        # Кожного разу, коли вміст scrollable_frame змінюється, область прокрутки canvas оновлюється
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Створюємо вікно всередині canvas на позиції (0,0) і прив'язуємо scrollable_frame до цього вікна
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Конфігуруємо canvas так, щоб прокрутка відбувалася за допомогою scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Упаковуємо canvas і scrollbar у контейнер
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class Application(tk.Tk):
    def __init__(self, signal):
        super().__init__()
        self.geometry("1800x900")
        self.title("wavelet")
        self.grid_columnconfigure(0, weight=10)  # Змінює розмір першого стовпця
        self.grid_columnconfigure(1, minsize=200)  # Встановлює мінімальний розмір для другого стовпця
        self.grid_rowconfigure(0, weight=100)  # Змінює розмір рядка
        self.model_NN = 'model1.h5'
        self.wavelet_transform = WaveletTransform(signal)
        self.scroll_frame = None
        self.wavelet_types = [wt for wt in pywt.wavelist(kind='discrete') if pywt.Wavelet(wt).family_name in ['Haar', 'Daubechies', 'Discrete Meyer', 'Symlets', 'Coiflets', 'Biorthogonal']]
        self.decomposition_levels = list(range(1, pywt.dwt_max_level(data_len=len(signal), filter_len=pywt.Wavelet('db2').dec_len) + 1))
        self.create_widgets()

    def create_widgets(self):
        # Scrollable Frame
        self.scroll_frame = ScrollableFrame(self)
        self.scroll_frame.grid(row=0, column=0, sticky='nsew')

        # Function Panel
        control_frame = tk.Frame(self)
        control_frame.grid(row=0, column=1, sticky='ns')

        wavelet_label = tk.Label(control_frame, text="Wavelet Type:")
        wavelet_label.pack(pady=10)

        self.wavelet_combobox = ttk.Combobox(control_frame, values=self.wavelet_types)
        self.wavelet_combobox.pack(pady=10)

        level_label = tk.Label(control_frame, text="Decomposition Level:")
        level_label.pack(pady=10)

        self.level_spinbox = ttk.Spinbox(control_frame, values=self.decomposition_levels)
        self.level_spinbox.pack(pady=10)

        recalculate_button = tk.Button(control_frame, text="Recalculate", command=self.recalculate)
        recalculate_button.pack(pady=10)

        # Adding input fields for window_size, split_ratio, and epochs
        label_ws = tk.Label(control_frame, text="Window size:")
        label_ws.pack(pady=5)
        self.window_size_entry = tk.Entry(control_frame)
        self.window_size_entry.pack(pady=10)
        self.window_size_entry.insert(0, "10")

        label_sr = tk.Label(control_frame, text="Split ratio:")
        label_sr.pack(pady=5)
        self.split_ratio_entry = tk.Entry(control_frame)
        self.split_ratio_entry.pack(pady=10)
        self.split_ratio_entry.insert(0, "0.7")

        label_e = tk.Label(control_frame, text="Epochs:")
        label_e.pack(pady=5)
        self.epochs_entry = tk.Entry(control_frame)
        self.epochs_entry.pack(pady=10)
        self.epochs_entry.insert(0, "20")

        button = tk.Button(control_frame, text="P_L_A NN", command=self.plot_last_approximation)  # Plot Last Approximation
        button.pack(pady=10)

        # Додаємо нову кнопку "PREDICT"
        predict_button = tk.Button(control_frame, text="PREDICT", command=self.predict_signal)
        predict_button.pack(pady=10)

        button_3d = tk.Button(control_frame, text="3D", command=self.plot_3d_graf_3)  # 3D graf
        button_3d.pack(pady=10)

        button_build = tk.Button(control_frame, text="build", command=self.build_graph)
        button_build.pack(pady=10)

        """Модель ARIMA використовується з порядком (5, 1, 0), що означає:
5 - порядок компоненти авторегресії (AR). Це кількість попередніх значень часового ряду, які використовуються 
для прогнозування наступного значення.
1 - порядок різниці (I). Це кількість разів, коли вхідні дані були "диференційовані", або коли з кожного значення 
було віднято попереднє значення.
0 - порядок компоненти ковзного середнього (MA). Це кількість термінів помилки в регресійній моделі."""

        # Adding input fields for ar, i, ma
        label_ar = tk.Label(control_frame, text="AR:")
        label_ar.pack(pady=5)
        self.ar_entry = tk.Entry(control_frame)
        self.ar_entry.pack(pady=10)
        self.ar_entry.insert(0, "5")

        label_i = tk.Label(control_frame, text="I:")
        label_i.pack(pady=5)
        self.i_entry = tk.Entry(control_frame)
        self.i_entry.pack(pady=10)
        self.i_entry.insert(0, "1")

        label_ma = tk.Label(control_frame, text="MA:")
        label_ma.pack(pady=5)
        self.ma_entry = tk.Entry(control_frame)
        self.ma_entry.pack(pady=10)
        self.ma_entry.insert(0, "0")

        button_A = tk.Button(control_frame, text="ARIMA", command=self.plot_arima)
        button_A.pack(pady=10)

        self.plot_figures()

    def plot_figures(self):
        # Clear the scrollable frame
        for widget in self.scroll_frame.scrollable_frame.winfo_children():
            widget.destroy()

        figs = [self.plot_signal(), self.plot_wavelet()]
        for i in range(len(self.wavelet_transform.coeffs) - 1):
            figs.append(self.plot_approximation(i + 1))
            figs.append(self.plot_details(i + 1))

        # Arrange plots into two columns
        for i in range(0, len(figs), 2):
            frame = tk.Frame(self.scroll_frame.scrollable_frame)
            frame.pack(fill=tk.BOTH, expand=True)
            self.create_canvas_and_toolbar(figs[i], frame)
            if i + 1 < len(figs):
                self.create_canvas_and_toolbar(figs[i + 1], frame)

    def recalculate(self):
        wavelet_type = self.wavelet_combobox.get()
        decomposition_level = int(self.level_spinbox.get())
        self.wavelet_transform = WaveletTransform(self.wavelet_transform.signal, wavelet=wavelet_type, level=decomposition_level)
        self.plot_figures()

    def plot_signal(self):
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(self.wavelet_transform.signal)
        ax.grid()
        ax.set_title("Signal[{}]".format(self.wavelet_transform.signal.size))
        return fig

    def plot_wavelet(self):
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        wave = pywt.Wavelet(self.wavelet_transform.wavelet.name)
        W_level = 10
        if W_level == 0:
            w = wave.wavefun()
        else:
            w = wave.wavefun(level=W_level)
        ax.plot(w[2], w[0], label='Скейлінг функція (φ)')
        ax.plot(w[2], w[1], label='Вейвлет функція (ψ)')
        ax.grid(True)
        ax.legend()
        ax.set_title("Wavelet ({})[{}]".format(self.wavelet_transform.wavelet.name, self.wavelet_transform.wavelet.dec_len))
        return fig

    def plot_approximation(self, level):
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        approximation = self.wavelet_transform.get_approximation(level)
        signal = self.wavelet_transform.signal
        metrics = Metrics(signal, approximation[:len(signal)])
        f1, f2, f3, f4, f5 = metrics.correlation_arr(), metrics.r2(), metrics.mse(), metrics.rmse(), metrics.mae()
        if approximation is not None:
            ax.plot(approximation)
            ax.text(0.01, 0.8, f'Correlation: {f1:.2f}\nR-squared: {f2:.2f}\nMSE: {f3:.2f}\nRMSE: {f4:.2f}\nMAE: {f5:.2f}', transform=ax.transAxes)
        ax.set_title("Approximation Level {}".format(level))
        ax.grid()
        return fig

    def plot_details(self, level):
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        details = self.wavelet_transform.get_details(level)
        if details is not None:
            ax.plot(details)
        ax.set_title("Detail Coefficients Level {}".format(level))
        # ax.set_title("Detail Coefficients Level {}[{}]".format(level, details.size))
        ax.grid()
        return fig

    def plot_last_approximation(self):
        start = time.time()
        last_level = len(self.wavelet_transform.coeffs) - 1
        if last_level == 0:
            x = self.wavelet_transform.signal  # Use the original signal if level is 0
        else:
            x = self.wavelet_transform.get_approximation(last_level)
        # Використання класу
        t = np.arange(0, x.size, 1)
        # Get the values from the input fields
        window_size = int(self.window_size_entry.get())
        split_ratio = float(self.split_ratio_entry.get())
        epochs = int(self.epochs_entry.get())
        predictor = TimeSeriesPredictor(x, window_size=window_size, split_ratio=split_ratio,
                                        epochs=epochs)  # Створення об'єкту класу TimeSeriesPredictor і передача йому сигналу x
        X, y = predictor.generate_data()  # Генерування вхідних та вихідних даних для моделі
        X_train, X_test, y_train, y_test, split_idx = predictor.split_data(X,
                                                                           y)  # Розділення даних на тренувальний та тестовий набори
        predictor.fit(X_train, y_train)  # Навчання моделі
        y_pred = predictor.predict(X_test)  # Прогнозування
        predictor.improve_and_save(X_train, y_train, 'my_model.h5')  # 'my_model.h5' | 'fnn1.h5'
        end = time.time()
        if x is not None:
            predictor.plot(t, y_train, y_test, y_pred, split_idx, end - start)
            plt.show()

    def predict_signal(self):
        start = time.time()
        last_level = len(self.wavelet_transform.coeffs) - 1

        window_size = int(self.window_size_entry.get())
        split_ratio = float(self.split_ratio_entry.get())
        # epochs = int(self.epochs_entry.get())

        if last_level == 0:
            x = self.wavelet_transform.signal  # Use the original signal if level is 0
        else:
            x = self.wavelet_transform.get_approximation(last_level)
        t = np.arange(0, x.size, 1)

        predictor = TimeSeriesPredictor(x, window_size=window_size, split_ratio=split_ratio, epochs=0)
        X, y = predictor.generate_data()
        X_train, X_test, y_train, y_test, split_idx = predictor.split_data(X, y)
        predictor.fit(X_train, y_train)
        y_pred = predictor.load_and_predict(X_test, 'my_model.h5')
        end = time.time()
        if x is not None:
            predictor.plot(t, y_train, y_test, y_pred, split_idx, end - start)
            plt.show()

    def plot_arima(self):
        last_level = len(self.wavelet_transform.coeffs) - 1
        if last_level == 0:
            x = self.wavelet_transform.signal
        else:
            x = self.wavelet_transform.get_approximation(last_level)
        # split_ratio = float(self.split_ratio_entry.get())

        ar_e = int(self.ar_entry.get())
        i_e = float(self.i_entry.get())
        ma_e = int(self.ma_entry.get())

        # herald = ARIMAPredictor(x, arima_order=(ar_e, i_e, ma_e), split_ratio=split_ratio)
        herald = ARIMAPredictor(x, arima_order=(ar_e, i_e, ma_e))
        herald.run()

    def plot_3d_graf_3(self):
        start = time.time()
        last_level = len(self.wavelet_transform.coeffs) - 1
        last_approximation = self.wavelet_transform.get_approximation(last_level)
        x = last_approximation
        numb_X = int(self.window_size_entry.get())  # window
        numb_Y = int(self.epochs_entry.get())  # epochs

        def calculate_correlation(i, j):
            predictor_c = TimeSeriesPredictor(x, window_size=i, split_ratio=0.7, epochs=j)
            X_c, y_c = predictor_c.generate_data()  # Генерування вхідних та вихідних даних для моделі
            X_train_c, X_test_c, y_train_c, y_test_c, split_idx_c = predictor_c.split_data(X_c, y_c)
            predictor_c.fit(X_train_c, y_train_c)  # Навчання моделі
            y_pred_c = predictor_c.predict(X_test_c)  # Прогнозування
            metr = Metrics(y_test_c, y_pred_c)
            return metr.correlation()

        # Паралельне виконання обчислення кореляції для кожного i, j
        results = Parallel(n_jobs=-1)(
            delayed(calculate_correlation)(i, j) for i in range(numb_Y) for j in range(numb_X))

        # Розпакування результатів у матрицю Z
        Z = np.array(results).reshape(numb_Y, numb_X)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np.arange(numb_X), np.arange(numb_Y))
        # Plot the surface
        surf = ax2.plot_surface(X, Y, Z, cmap='plasma')  # ## 1 viridis
        # Add labels
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('Epochs')
        ax2.set_zlabel('Correlation')
        end = time.time()
        print(f'time: {end - start:.5f}')
        # Add a color bar which maps values to colors.
        fig2.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def wavelet_metrics_analysis_1(self, wavelet_list, decomposition_level, window_size, split_ratio, epochs):
        metrics_results = []
        for wavelet in wavelet_list:
            self.wavelet_transform = WaveletTransform(self.wavelet_transform.signal, wavelet=wavelet, level=decomposition_level)
            x = self.wavelet_transform.get_approximation(decomposition_level)
            if x is not None:
                predictor = TimeSeriesPredictor(x, window_size=window_size, split_ratio=split_ratio, epochs=epochs)
                X, y = predictor.generate_data()
                X_train, X_test, y_train, y_test, split_idx = predictor.split_data(X, y)
                predictor.fit(X_train, y_train)
                y_pred = predictor.predict(X_test)
                metrics = Metrics(y_test, y_pred)
                metrics_results.append([metrics.correlation(), metrics.r2()])  # metrics.mse(), metrics.rmse(), metrics.mae()
        return metrics_results

    def wavelet_metrics_analysis(self, wavelet_list, decomposition_level, window_size, split_ratio, epochs):
        metrics_results = []
        for wavelet in wavelet_list:
            self.wavelet_transform = WaveletTransform(self.wavelet_transform.signal, wavelet=wavelet,
                                                      level=decomposition_level)
            x = self.wavelet_transform.get_approximation(decomposition_level)
            if x is not None:
                predictor = TimeSeriesPredictor(x, window_size=window_size, split_ratio=split_ratio, epochs=0)
                X, y = predictor.generate_data()
                X_train, X_test, y_train, y_test, split_idx = predictor.split_data(X, y)

                y_pred = predictor.load_and_predict(X_test, 'my_model.h5')

                metrics = Metrics(y_test, y_pred)
                # metrics_results.append([metrics.correlation(), metrics.r2()])
                # metrics_results.append([metrics.mse(), metrics.rmse(), metrics.mae()])
                # metrics.correlation(), metrics.r2(), metrics.mse(), metrics.rmse(), metrics.mae()
                metrics_results.append([metrics.correlation(), metrics.r2(), metrics.mse(), metrics.rmse(), metrics.mae()])
                print(metrics.correlation())
        return metrics_results

    def build_graph(self):
        wavelet_list = ['db' + str(i) for i in range(1, 39)]  # coif 1-17[1-18] | sym 2-20[2-21] | db 1-38[1-39]
        decomposition_level = len(self.wavelet_transform.coeffs) - 1
        window_size = int(self.window_size_entry.get())
        split_ratio = float(self.split_ratio_entry.get())
        epochs = int(self.epochs_entry.get())
        metrics_results = self.wavelet_metrics_analysis(wavelet_list, decomposition_level, window_size, split_ratio, epochs)
        metrics_results = np.array(metrics_results).T
        print(metrics_results)
        metr_name = np.array(['Correlation', 'R-squared', 'MSE', 'RMSE', 'MAE'])
        fig, ax = plt.subplots()
        for i, metric in enumerate(metrics_results):
            ax.plot(wavelet_list, metric, label=f'{metr_name[i+0]}')  # 0 or 2
        ax.legend()
        plt.grid(True)
        plt.show()

    def create_canvas_and_toolbar(self, fig, frame):
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


class CustomTable(Table):
    def __init__(self, parent=None, **kwargs):  # Конструктор класу, приймає параметри parent та kwargs
        Table.__init__(self, parent, **kwargs)  # Виклик конструктора батьківського класу з переданими параметрами

    def plot_graph(self, col_name, start_row, end_row):
        if col_name is None or col_name == '':
            return

        data = self.model.df[col_name].iloc[start_row:end_row]  # Змінено на iloc з використанням start_row та end_row
        app = Application(data.values)
        app.mainloop()

    def get_current_column_name(self):  # Метод отримання імені виділеного стовпця
        col_index = self.getSelectedColumn()  # Отримання індексу виділеного стовпця
        if col_index is None or col_index == '':  # Перевірка, чи отримано індекс стовпця
            return None
        return self.model.df.columns[col_index]  # Повернення імені стовпця за його індексом


def show_plot_window(table):
    def on_submit():
        start_row = int(entry_start_row.get())
        end_row = int(entry_end_row.get())
        col_name = table.get_current_column_name()
        chosen_style = style_var.get()
        style.use(chosen_style)
        if col_name is None or col_name == '':
            return
        table.plot_graph(col_name, start_row, end_row)

    window = tk.Toplevel()
    window.title("Plot Graph")

    label_start_row = tk.Label(window, text="Enter the starting row number:")
    label_start_row.grid(row=0, column=0)

    entry_start_row = tk.Entry(window)
    entry_start_row.grid(row=0, column=1)

    label_end_row = tk.Label(window, text="Enter the ending row number:")
    label_end_row.grid(row=1, column=0)

    entry_end_row = tk.Entry(window)
    entry_end_row.grid(row=1, column=1)

    style_label = tk.Label(window,
                           text="Choose a plot style:")  # Додаємо випадаючий список з доступними стилями графіків
    style_label.grid(row=3, column=0)

    style_var = tk.StringVar(window)  # Створення змінної StringVar для зберігання обраного стилю графіка
    style_var.set(available_styles[0])  # Встановлення значення за замовчуванням (перший стиль у списку)

    style_menu = tk.OptionMenu(window, style_var, *available_styles)  # Створення випадаючого списку зі стилями графіків
    style_menu.grid(row=3, column=1)

    submit_button = tk.Button(window, text="Submit", command=on_submit)  # Створення кнопки "Submit"
    submit_button.grid(row=4, column=0)  # Розміщення кнопки в вікні

    destroy_button = tk.Button(window, text="Destroy", command=window.destroy)
    destroy_button.grid(row=4, column=2)


def main():
    # data = pd.DataFrame()
    data = pd.DataFrame({"A": range(10), "B": range(10, 20)})  # Створення прикладу DataFrame

    root = tk.Tk()  # Створення головного вікна програми
    root.title("Custom Table with Plotting")  # Встановлення заголовка головного вікна

    frame1 = tk.Frame(root)  # Створення рамки (контейнера) для таблиці
    frame1.pack(fill='both', expand=True)  # Розміщення рамки та забезпечення зміни розміру разом з вікном

    frame2 = tk.Frame(root)  # Створення рамки для кнопки
    frame2.pack(side='left')  # Розміщення рамки з кнопкою знизу вікна

    table = CustomTable(frame1, dataframe=data, showtoolbar=True, showstatusbar=True)  # Створення об'єкту таблиці
    table.show()  # Відображення таблиці в рамці

    plot_button = tk.Button(frame2, text="Plot Graph",  command=lambda: show_plot_window(table))  # Створення кнопки "Plot Graph"
    plot_button.grid(row=0, column=0, sticky='w')  # Розміщення кнопки в рамці

    root.mainloop()  # Запуск основного циклу обробки подій головного вікна


if __name__ == "__main__":
    main()
