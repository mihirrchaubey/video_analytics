@echo off
REM Run both backend and frontend services

echo.
echo Starting Video Analytics...
echo.

REM Check if venv exists
if not exist "venv" (
    echo Error: Virtual environment not found. Run setup.bat first
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Create storage directories
if not exist "storage\videos" mkdir storage\videos
if not exist "storage\frames" mkdir storage\frames
if not exist "storage\chroma" mkdir storage\chroma

echo.
echo Starting services...
echo.
echo FastAPI Backend: http://localhost:8000
echo Streamlit UI:    http://localhost:8501
echo API Docs:        http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop services
echo.

REM Start backend in new window
start "Video Analytics - Backend" cmd /k python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

REM Wait a bit for backend to start
timeout /t 3 /nobreak

REM Start frontend
streamlit run app\ui.py --server.port 8501 --server.address 0.0.0.0
