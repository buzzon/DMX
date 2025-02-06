import pyaudio
import numpy as np
import scipy.fftpack

# Конфигурация аудио
FORMAT = pyaudio.paInt16
CHANNELS = 2  # Стерео захват для системного звука
RATE = 44100
CHUNK = 2048  # Размер блока данных

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Поиск loopback-устройства
def find_loopback_device():
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if "Stereo Mix" in dev_info["name"] and dev_info["maxInputChannels"] > 0:
            return i
    return None

# Получаем ID loopback-устройства
device_id = find_loopback_device()

if device_id is None:
    print("Loopback устройство не найдено! Убедитесь, что:")
    print("1. Включена запись с 'Stereo Mix' в настройках звука Windows")
    print("2. Установлены последние аудио-драйверы")
    p.terminate()
    exit()

# Открываем поток в loopback-режиме
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
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
    freqs = scipy.fftpack.fftfreq(len(fft), 1.0 / RATE)
    
    # Фильтрация частот
    mask = freqs > 0
    fft = np.abs(fft[mask])
    freqs = freqs[mask]
    
    # Расчет энергии в диапазонах
    low = np.sum(fft[(freqs >= 20) & (freqs < 250)])
    mid = np.sum(fft[(freqs >= 250) & (freqs < 4000)])
    high = np.sum(fft[(freqs >= 4000) & (freqs < 20000)])
    
    return low, mid, high

try:
    print("Анализатор запущен. Нажмите Ctrl+C для остановки...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        low, mid, high = analyze_frequencies(data)
        
        # Нормализация значений
        total = low + mid + high
        if total > 0:
            low_pct = low / total * 100
            mid_pct = mid / total * 100
            high_pct = high / total * 100
            print(f"\rНизкие: {low_pct:5.1f}% | Средние: {mid_pct:5.1f}% | Высокие: {high_pct:5.1f}%", end="")

except KeyboardInterrupt:
    print("\nОстановка анализатора...")
    stream.stop_stream()
    stream.close()
    p.terminate()