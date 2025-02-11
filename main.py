import pyaudio
import numpy as np
import scipy.fftpack

from scipy.interpolate import interp1d

# Конфигурация аудио
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Стерео захват для системного звука

SAMPLE_RATE = 44100  # Частота дискретизации
CHUNK = 1024  # Размер блока данных

# Границы частотных диапазонов (Гц)
LOW_CUTOFF = 20
MID_CUTOFF = 500
HIGH_CUTOFF = 4000

SERIAL_PORT = 'COM4'

lmax = 0
mmax = 0
hmax = 0

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Поиск loopback-устройства
def find_loopback_device():
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print ("-  ", (dev_info["name"]), i)

# Получаем ID loopback-устройства
device_id =  1

# Открываем поток в loopback-режиме
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=device_id,
    frames_per_buffer=CHUNK
)

def analyze_frequencies(data):
    # Конвертация данных в numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # Нормализация и преобразование в моно
    if CHANNELS == 2:
        audio_data = audio_data.reshape((-1, 2))
        audio_data = audio_data.mean(axis=1)
    
    # Применение окна Ханна для уменьшения артефактов
    window = np.hanning(len(audio_data))
    audio_data = audio_data * window
    
    # Выполнение FFT
    fft = scipy.fftpack.fft(audio_data)
    frequencies =  scipy.fftpack.fftfreq(len(audio_data), d=1/SAMPLE_RATE)
    magnitudes = np.abs(fft) / len(fft)
    
    # Разделение частот на диапазоны
    low_mask = (frequencies >= LOW_CUTOFF) & (frequencies < MID_CUTOFF)
    mid_mask = (frequencies >= MID_CUTOFF) & (frequencies < HIGH_CUTOFF)
    high_mask = frequencies >= HIGH_CUTOFF

    # Расчет средних значений для каждого диапазона
    low_level = np.mean(magnitudes[low_mask]) if any(low_mask) else 0
    mid_level = np.mean(magnitudes[mid_mask]) if any(mid_mask) else 0
    high_level = np.mean(magnitudes[high_mask]) if any(high_mask) else 0

    
    return low_level, mid_level, high_level


def mapFromTo(x,a,b,c,d):
   y=(x-a)/(b-a)*(d-c)+c
   return y

try:
    print("Анализатор запущен. Нажмите Ctrl+C для остановки...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        low, mid, high = analyze_frequencies(data)

        lmax = max(lmax, low)
        mmax = max(mmax, mid)
        hmax = max(hmax, high)

        low_dmx = mapFromTo(low, 0, lmax, 0, 255)
        mid_dmx = mapFromTo(mid, 0, mmax, 0, 255)
        high_dmx = mapFromTo(high, 0, hmax, 0, 255)

        # Нормализация значений
        total = low + mid + high
        if total > 0:
            low_pct = low 
            mid_pct = mid 
            high_pct = high 
            # print(f"\rНизкие: {low_pct:5.1f}\t|{lmax:5.1f} \t| Средние: {mid_pct:5.1f}\t|{mmax:5.1f} \t| Высокие: {high_pct:5.1f}\t|{hmax:5.1f}", end="")
            print(f'\r{low_dmx:5.1f} \t{mid_dmx:5.1f} \t{high_dmx:5.1f}', end='', flush=True)

except KeyboardInterrupt:
    print("\nОстановка анализатора...")
    stream.stop_stream()
    stream.close()
    p.terminate()