# app.py
# Filtros digitales interactivos con Streamlit: soporte para audio, archivos y grabación
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.io import wavfile
import pandas as pd
import io
import sounddevice as sd
import tempfile
import time
from pydub import AudioSegment

# Configuración de la página
st.set_page_config(page_title="🎧 Filtros Digitales con Audio", layout="wide")
st.title("🎧 Filtros Digitales Interactivos")
st.markdown("Diseña filtros y aplícalos a señales sintéticas, archivos `.wav` o `.csv`, o graba desde tu micrófono.")

# ================================================
# SIDEBAR: Selección de modo
# ================================================
st.sidebar.header("🔧 Modo de entrada")
modo = st.sidebar.radio(
    "Selecciona la fuente de la señal",
    ["Señal sintética", "Cargar archivo (.wav o .csv)", "Grabar desde micrófono"]
)

# Variables globales
senal_ruidosa = None
fs = 1000  # Frecuencia de muestreo por defecto
t = None

# ================================================
# 1. Señal sintética
# ================================================
if modo == "Señal sintética":
    st.sidebar.subheader("Parámetros de la señal")
    fs = st.sidebar.slider("Frecuencia de muestreo (Hz)", 500, 2000, 1000)
    T = st.sidebar.slider("Duración (s)", 1.0, 5.0, 2.0)
    snr_db = st.sidebar.slider("SNR (dB)", 5, 30, 15)

    frecuencias = st.sidebar.multiselect(
        "Frecuencias componentes (Hz)",
        [1, 5, 10, 20, 50, 60, 100, 120, 200, 300],
        default=[5, 50, 120]
    )

    t = np.linspace(0, T, int(fs * T), endpoint=False)
    senal_limpia = np.sum([np.sin(2 * np.pi * f * t) for f in frecuencias], axis=0)
    senal_poder = np.mean(senal_limpia ** 2)
    ruido_poder = senal_poder / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(ruido_poder), senal_limpia.shape)
    senal_ruidosa = senal_limpia + ruido

# ================================================
# 2. Cargar archivo .wav o .csv
# ================================================
elif modo == "Cargar archivo (.wav o .csv)":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo .wav o .csv", type=["wav", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".wav"):
                fs, audio_data = wavfile.read(uploaded_file)
                # Convertir a mono si es estéreo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = audio_data.astype(np.float64)
                # Normalizar
                audio_data = audio_data / np.max(np.abs(audio_data))
                senal_ruidosa = audio_data
                t = np.linspace(0, len(audio_data) / fs, len(audio_data))

                st.success(f"✅ Audio cargado: {uploaded_file.name} | fs = {fs} Hz | Duración = {len(audio_data)/fs:.2f} s")

            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if df.shape[1] < 2:
                    st.error("El archivo CSV debe tener al menos dos columnas: tiempo y señal.")
                else:
                    t = df.iloc[:, 0].values
                    senal_ruidosa = df.iloc[:, 1].values
                    fs = int(1 / (t[1] - t[0]))  # Estimar fs
                    st.success(f"✅ CSV cargado: {uploaded_file.name} | fs estimado = {fs} Hz")

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    else:
        st.info("Por favor, sube un archivo .wav o .csv.")
        st.stop()

# ================================================
# 3. Grabar desde micrófono
# ================================================
elif modo == "Grabar desde micrófono":
    st.sidebar.subheader("Grabación de audio")
    duration = st.sidebar.slider("Duración de grabación (s)", 1, 10, 3)
    fs = st.sidebar.number_input("Frecuencia de muestreo", value=44100, step=1000)

    if st.sidebar.button("🎙️ Grabar ahora"):
        try:
            with st.spinner("Grabando..."):
                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
                sd.wait()  # Esperar a que termine
                senal_ruidosa = audio_data.flatten()
                t = np.linspace(0, duration, len(senal_ruidosa))
                st.success("✅ Grabación finalizada.")
        except Exception as e:
            st.error(f"Error en la grabación: {e}")
            st.stop()

# Si no se tiene señal, detener
if senal_ruidosa is None:
    st.warning("No se ha generado ninguna señal. Por favor, selecciona una opción válida.")
    st.stop()

# ================================================
# Diseño del filtro
# ================================================
st.sidebar.header("🎛️ Diseño del Filtro")
filtro_tipo = st.sidebar.selectbox("Tipo de filtro", ["Pasa Bajos", "Pasa Altos", "Pasa Banda"])
filtro_clase = st.sidebar.selectbox("Tipo de filtro digital", ["FIR", "IIR"])

nyq = 0.5 * fs
b, a = None, None

if filtro_tipo == "Pasa Bajos":
    lowcut = st.sidebar.slider("Frecuencia de corte (Hz)", 1, min(400, int(nyq)), 30)
    lowcut_norm = lowcut / nyq
    if filtro_clase == "IIR":
        b, a = signal.butter(4, lowcut_norm, btype='low')
    else:
        b = signal.firwin(65, lowcut, fs=fs, window='hamming')
        a = 1

elif filtro_tipo == "Pasa Altos":
    highcut = st.sidebar.slider("Frecuencia de corte (Hz)", 1, min(400, int(nyq)), 40)
    highcut_norm = highcut / nyq
    if filtro_clase == "IIR":
        b, a = signal.cheby1(4, 1, highcut_norm, btype='high')
    else:
        b = signal.firwin(65, highcut, fs=fs, pass_zero=False, window='hamming')
        a = 1

else:  # Pasa Banda
    low_band = st.sidebar.slider("Frecuencia inferior (Hz)", 1, 200, 40)
    high_band = st.sidebar.slider("Frecuencia superior (Hz)", low_band + 10, 500, 60)
    if filtro_clase == "IIR":
        b, a = signal.butter(4, [low_band / nyq, high_band / nyq], btype='band')
    else:
        b = signal.firwin(65, [low_band, high_band], fs=fs, pass_zero=False, window='hamming')
        a = 1

# ================================================
# Aplicar filtro
# ================================================
senal_filtrada = signal.filtfilt(b, a, senal_ruidosa)

# Normalizar para evitar saturación
senal_filtrada = senal_filtrada / np.max(np.abs(senal_filtrada))

# ================================================
# Gráficas
# ================================================
st.subheader("📊 Resultados del filtrado")

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# 1. Tiempo: antes vs después
axs[0, 0].plot(t[:600], senal_ruidosa[:600], label="Original", alpha=0.7)
axs[0, 0].plot(t[:600], senal_filtrada[:600], label="Filtrada", color='red', linewidth=1.5)
axs[0, 0].set_title("Señal en el tiempo")
axs[0, 0].set_xlabel("Tiempo [s]")
axs[0, 0].set_ylabel("Amplitud")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. FFT
N = len(senal_ruidosa)
Y_orig = fft.fft(senal_ruidosa)
Y_filt = fft.fft(senal_filtrada)
xf = fft.fftfreq(N, 1/fs)[:N//2]
axs[0, 1].plot(xf, 2.0/N * np.abs(Y_orig[:N//2]), label="Original", alpha=0.8)
axs[0, 1].plot(xf, 2.0/N * np.abs(Y_filt[:N//2]), label="Filtrada", color='red')
axs[0, 1].set_title("Espectro de frecuencia")
axs[0, 1].set_xlabel("Frecuencia [Hz]")
axs[0, 1].set_ylabel("Magnitud")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Respuesta del filtro
w, h = signal.freqz(b, a, fs=fs)
axs[1, 0].plot(w, 20 * np.log10(np.abs(h) + 1e-10), color='purple')
axs[1, 0].set_title(f"Respuesta en frecuencia - {filtro_tipo} ({filtro_clase})")
axs[1, 0].set_xlabel("Frecuencia [Hz]")
axs[1, 0].set_ylabel("Ganancia [dB]")
axs[1, 0].axvline(fs/2, color='r', linestyle='--', alpha=0.5, label="Nyquist")
axs[1, 0].grid(True)
axs[1, 0].legend()

# 4. Comparación señal limpia (solo si es sintética)
if modo == "Señal sintética":
    senal_limpia = senal_ruidosa - np.random.normal(0, np.std(senal_ruidosa - senal_limpia), senal_ruidosa.shape)
    axs[1, 1].plot(t[:600], senal_limpia[:600], label="Ideal", linestyle='--', alpha=0.7)
else:
    axs[1, 1].plot(t[:600], senal_ruidosa[:600], label="Entrada", alpha=0.7)
axs[1, 1].plot(t[:600], senal_filtrada[:600], label="Salida", color='green')
axs[1, 1].set_title("Comparación entrada vs salida")
axs[1, 1].set_xlabel("Tiempo [s]")
axs[1, 1].set_ylabel("Amplitud")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
st.pyplot(fig)

# ================================================
# Reproducir audio filtrado (solo si hay audio)
# ================================================
if st.button("🔊 Reproducir audio filtrado"):
    try:
        sd.play(senal_filtrada, fs)
        st.info("Reproduciendo...")
        sd.wait()
        st.success("✅ Reproducción finalizada.")
    except Exception as e:
        st.error(f"Error al reproducir: {e}")

# ================================================
# Descargar señal filtrada
# ================================================
csv_data = io.StringIO()
pd.DataFrame({"tiempo": t, "senal_filtrada": senal_filtrada}).to_csv(csv_data, index=False)
st.download_button(
    label="⬇️ Descargar señal filtrada (.csv)",
    data=csv_data.getvalue(),
    file_name="senal_filtrada.csv",
    mime="text/csv"
)

# Opcional: descargar como WAV (si es audio)
if len(senal_filtrada) > 0 and modo != "Señal sintética":
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wavfile.write(tmpfile.name, fs, (senal_filtrada * 32767).astype(np.int16))
        with open(tmpfile.name, "rb") as f:
            wav_bytes = f.read()
    st.download_button(
        label="⬇️ Descargar como audio (.wav)",
        data=wav_bytes,
        file_name="audio_filtrado.wav",
        mime="audio/wav"
    )