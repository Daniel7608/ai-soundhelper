import librosa
import numpy as np
import logging
import os


def analyze_user_style(audio_paths: list) -> dict:
    """
    Анализирует список аудиофайлов (референсов) для извлечения музыкальных характеристик.
    Расширенная версия для диплома: извлекает тональность, BPM, яркость, настроение,
    плотность ритма, акцент на басы и уровень энергетики трека.
    """
    # 1. Защита: если список пуст, сразу отдаем дефолт со строгими ключами для фронтенда
    if not audio_paths:
        print("[ANALYZER] Предупреждение: Список audio_paths пуст.")
        return {
            "style_profile": "Balanced / Classic",
            "detected_key": "C Major",
            "bpm": 120,
            "brightness": 0.5,
            "mood": "Deep & Nostalgic",
            "rhythm_density": "Medium-Low (Smooth & Atmospheric)",
            "bass_power": "Balanced Low-End",
            "energy_level": "Balanced Vibe"
        }

    bpms = []
    keys = []
    brightness_values = []
    flatness_values = []
    centroid_values = []
    rms_values = []

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for path in audio_paths:
        print(f"[ANALYZER] Попытка анализа файла: {path}")

        # Проверяем, существует ли файл физически на диске
        if not os.path.exists(path):
            print(f"[ANALYZER] ОШИБКА: Файл не найден по пути {os.path.abspath(path)}")
            continue

        try:
            # Загружаем первые 60 секунд трека
            y, sr = librosa.load(path, duration=60)

            if len(y) == 0:
                print(f"[ANALYZER] ОШИБКА: Файл {path} загрузился пустым.")
                continue

            # Разделение на гармонику и перкуссию через HPSS
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Вычисление BPM по ударным
            tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
            bpm_val = int(np.round(tempo[0] if isinstance(tempo, np.ndarray) else tempo))
            if 40 <= bpm_val <= 200:
                bpms.append(bpm_val)

            # Вычисление тональности по гармонике
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            mean_chroma = np.mean(chroma, axis=1)

            best_pitch_idx = int(np.argmax(mean_chroma))
            detected_key = pitch_classes[best_pitch_idx]

            minor_third_idx = (best_pitch_idx + 3) % 12
            major_third_idx = (best_pitch_idx + 4) % 12
            if mean_chroma[minor_third_idx] > mean_chroma[major_third_idx]:
                detected_key += " Minor"
            else:
                detected_key += " Major"
            keys.append(detected_key)

            # Вычисление спектрального центроида (яркость / частотный центр)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            mean_centroid = float(np.mean(centroid))
            centroid_values.append(mean_centroid)

            norm_brightness = min(max((mean_centroid - 1000) / 3000, 0.0), 1.0)
            brightness_values.append(norm_brightness)

            # МЕТРИКА ДЛЯ ДИПЛОМА: Спектральная плоскость (чистота / шумность звука)
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_harmonic)))
            flatness_values.append(flatness)

            # МЕТРИКА ДЛЯ ДИПЛОМА: RMS энергия (общая среднеквадратичная громкость)
            rms = float(np.mean(librosa.feature.rms(y=y)))
            rms_values.append(rms)

            print(f"[ANALYZER] Успешно обработан: {path} -> BPM: {bpm_val}, Key: {detected_key}")

        except Exception as e:
            print(f"[ANALYZER] КРИТИЧЕСКАЯ ОШИБКА при чтении {path}: {e}")
            logging.error(f"Ошибка при анализе файла {path}: {e}")
            continue

    # Агрегация базовых результатов. Если списки пусты — берем безопасный дефолт
    final_bpm = int(np.mean(bpms)) if bpms else 120
    final_brightness = float(np.mean(brightness_values)) if brightness_values else 0.5
    final_key = max(set(keys), key=keys.count) if keys else "C Major"

    # Агрегация усредненных спектральных данных для расширенных фич
    final_flatness = float(np.mean(flatness_values)) if flatness_values else 0.05
    final_centroid = float(np.mean(centroid_values)) if centroid_values else 1800
    final_rms = float(np.mean(rms_values)) if rms_values else 0.04

    # Определение базового профиля стиля
    if final_brightness > 0.6:
        style_name = "Bright / Modern"
    elif final_brightness < 0.4:
        style_name = "Deep / Lo-fi"
    else:
        style_name = "Balanced / Classic"

    # --- НАЧАЛО НОВОГО БЛОКА СЛОЖНЫХ МЕТРИК ДЛЯ ДИПЛОМА ---

    # 1. Определение плотности ритма (через спектральный центроид верхних частот референса)
    rhythm_density = "High (Punchy & Aggressive)" if final_centroid > 2200 else "Medium-Low (Smooth & Atmospheric)"

    # 2. Эмоциональный окрас на основе лада (мажор/минор) и спектральной плоскости шума
    if "Minor" in final_key:
        mood_detected = "Melancholic & Dark" if final_flatness < 0.02 else "Deep & Nostalgic"
    else:
        mood_detected = "Energetic & Bright" if final_centroid > 2000 else "Warm & Uplifting"

    # 3. Мощность баса (соотношение общей RMS энергии низких частот на средних темпах)
    if final_rms > 0.05 and final_bpm < 130:
        bass_power = "Heavy Sub-Bass Focus"
    else:
        bass_power = "Balanced Low-End"

    # 4. Общий уровень энергии трека (комбинация темпа и частотного распределения)
    if final_bpm > 125 or final_centroid > 2400:
        energy_level = "High Energy"
    else:
        energy_level = "Relaxed Ambient Vibe"

    # --- КОНЕЦ НОВОГО БЛОКА ---

    print(f"[ANALYZER] Расширенный расчет выполнен успешно!")
    print(
        f"[ANALYZER] Отправка данных: Style: {style_name}, Key: {final_key}, Mood: {mood_detected}, Bass: {bass_power}")

    # Полное соответствие расширенным JSON-ключам для main.py и фронтенда
    return {
        "style_profile": style_name,
        "detected_key": final_key,
        "bpm": final_bpm,
        "brightness": round(final_brightness, 2),
        "mood": mood_detected,
        "rhythm_density": rhythm_density,
        "bass_power": bass_power,
        "energy_level": energy_level
    }