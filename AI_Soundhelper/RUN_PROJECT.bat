@echo off
title AI Soundhelper Runner
color 0A
echo ========================================
echo    AI Soundhelper - Launch & Setup
echo ========================================
echo.

:: 1. Проверка Python
echo [1/5] Checking Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [OK] Python %PYTHON_VER% found
echo.

:: 2. Проверка pip
echo [2/5] Checking pip...
pip --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found!
    pause
    exit /b 1
)
echo [OK] pip found
echo.

:: 3. Проверка и установка зависимостей
echo [3/5] Checking dependencies...
pip show fastapi > nul 2>&1
if errorlevel 1 (
    echo [!] Dependencies missing. Installing...
    echo This may take 2-3 minutes...
    pip install fastapi uvicorn torch torchaudio transformers librosa scipy numpy
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already present
)
echo.

:: 4. Создание необходимых папок (если их нет)
echo [4/5] Creating required folders...
if not exist "backend\uploads" mkdir "backend\uploads"
if not exist "backend\results" mkdir "backend\results"
if not exist "backend" (
    echo [ERROR] Backend folder not found!
    pause
    exit /b 1
)
echo [OK] Folders ready
echo.

:: 5. Запуск бэкенда и фронтенда
echo [5/5] Starting services...
echo.

:: Запуск бэкенда в отдельном окне
cd backend
start "AI Soundhelper - Backend" cmd /k "python main.py"

:: Ждём 4 секунды для инициализации сервера
timeout /t 4 /nobreak > nul

:: Открытие фронтенда (проверяем, есть ли файл)
cd ../frontend
if exist "index.html" (
    start index.html
    echo [OK] Frontend opened in browser
) else (
    echo [WARN] index.html not found in frontend folder
    echo Please open frontend/index.html manually
)

echo.
echo ========================================
echo    SYSTEM IS RUNNING!
echo ========================================
echo Backend:  http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
echo [INFO] Do NOT close the backend window!
echo [INFO] Press any key to close this launcher...
pause > nul