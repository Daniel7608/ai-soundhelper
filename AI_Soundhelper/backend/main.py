import os
import uuid
import logging
import shutil
import torch
import sqlite3
import scipy.io.wavfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from analyzer import analyze_user_style

# --- Инициализация Базы Данных ---
def init_db():
    conn = sqlite3.connect('ai_soundhelper.db')
    cursor = conn.cursor()
    # Таблица для профилей стиля
    cursor.execute('''CREATE TABLE IF NOT EXISTS profiles 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, style TEXT, key TEXT, brightness REAL)''')
    # Таблица для сохраненных треков (обязательно с title)
    cursor.execute('''CREATE TABLE IF NOT EXISTS tracks 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, filename TEXT, genre TEXT, mood TEXT, url TEXT)''')
    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="AI Soundhelper API")

# Разрешаем CORS для связи с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание необходимых папок
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/results", StaticFiles(directory="results"), name="results")

# --- Загрузка MusicGen Small (Оптимально для GPU 6GB) ---
logging.basicConfig(level=logging.INFO)
logging.info("Загрузка модели MusicGen Small...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Используем small версию, чтобы избежать OutOfMemory на твоей видеокарте
synthesizer = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    saved_paths = []
    for file in files:
        path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(path)

    # Вызываем наш обновленный анализатор
    analysis_result = analyze_user_style(saved_paths)

    # Собираем все новые метрики для фронтенда
    return {
        "style_profile": analysis_result.get("style_profile", "Deep / Lo-fi"),
        "detected_key": analysis_result.get("detected_key", "D# Minor"),
        "bpm": analysis_result.get("bpm", 120),
        "brightness": analysis_result.get("brightness", 0.5),
        "mood": analysis_result.get("mood", "Deep / Lo-fi"),
        "rhythm_density": analysis_result.get("rhythm_density", "Medium-Low"),
        "bass_power": analysis_result.get("bass_power", "Balanced Low-End"),
        "energy_level": analysis_result.get("energy_level", "Relaxed Vibe")
    }

@app.post("/api/generate")
async def generate(
        prompt: str = Form(...), genre: str = Form(...), bpm: int = Form(...),
        duration: int = Form(...), key: str = Form(...), mood: str = Form(...)
):
    # Детальный промпт для нейросети
    full_prompt = f"{genre} style, {mood}, {prompt}, {bpm} BPM, key of {key}, studio quality, masterpiece"

    try:
        logging.info(f"Начало генерации трека: {full_prompt}")
        # 1 секунда звука ~ 50 токенов
        max_tokens = duration * 50

        output = synthesizer(full_prompt, forward_params={"max_new_tokens": max_tokens})

        filename = f"gen_{uuid.uuid4()}.wav"
        filepath = os.path.join("results", filename)

        # Сохранение файла через scipy (не требует установки ffmpeg в систему)
        scipy.io.wavfile.write(filepath, rate=output["sampling_rate"], data=output["audio"])

        return {
            "url": f"http://localhost:8000/results/{filename}",
            "filename": filename
        }
    except Exception as e:
        logging.error(f"Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ТВОЁ НАЗВАНИЕ: save_track
@app.post("/api/save_track")
async def save_track(title: str = Form(...), filename: str = Form(...), genre: str = Form(...), mood: str = Form(...),
                     url: str = Form(...)):
    conn = sqlite3.connect('ai_soundhelper.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tracks (title, filename, genre, mood, url) VALUES (?, ?, ?, ?, ?)",
                   (title, filename, genre, mood, url))
    conn.commit()
    conn.close()
    return {"status": "saved"}

# ТВОЁ НАЗВАНИЕ: track
@app.get("/api/tracks")
async def get_track():
    conn = sqlite3.connect('ai_soundhelper.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tracks ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/save_profile")
async def save_profile(name: str = Form(...), style: str = Form(...), key: str = Form(...),
                       brightness: float = Form(...)):
    conn = sqlite3.connect('ai_soundhelper.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO profiles (name, style, key, brightness) VALUES (?,?,?,?)",
                   (name, style, key, brightness))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/profiles")
async def get_profiles():
    conn = sqlite3.connect('ai_soundhelper.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM profiles ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.delete("/api/delete_profile/{profile_id}")
async def delete_profile(profile_id: int):
    conn = sqlite3.connect('ai_soundhelper.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.delete("/api/delete_track/{track_id}")
async def delete_track(track_id: int):
    conn = sqlite3.connect('ai_soundhelper.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM tracks WHERE id = ?", (track_id,))
    row = cursor.fetchone()

    if row:
        file_path = os.path.join("results", row['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)

        cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
        conn.commit()

    conn.close()
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)