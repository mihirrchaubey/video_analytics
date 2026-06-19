@echo off
REM Video Analytics Setup Script for Windows

echo.
echo ============================================
echo    Video Analytics - Setup and Installation
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [OK] Pip upgraded

REM Install dependencies
echo.
echo Installing dependencies (this may take several minutes)...
pip install -r requirements.txt
echo [OK] Dependencies installed

REM Create .env file
if not exist ".env" (
    echo.
    echo Creating .env file from template...
    copy .env.example .env
    echo [OK] .env file created
) else (
    echo [INFO] .env file already exists
)

REM Create storage directories
echo.
echo Creating storage directories...
if not exist "storage\videos" mkdir storage\videos
if not exist "storage\frames" mkdir storage\frames
if not exist "storage\chroma" mkdir storage\chroma
echo [OK] Storage directories ready

echo.
echo ============================================
echo [OK] Setup Complete!
echo ============================================
echo.
echo Next steps:
echo.
echo Option 1: Start both services
echo   run_app.bat
echo.
echo Option 2: Start manually in separate windows
echo.
echo Terminal 1 - Start FastAPI backend:
echo   venv\Scripts\activate.bat
echo   python -m uvicorn app.main:app --reload
echo.
echo Terminal 2 - Start Streamlit frontend:
echo   venv\Scripts\activate.bat
echo   streamlit run app\ui.py
echo.
echo Then open: http://localhost:8501
echo.
echo Option 3: Run test pipeline
echo   venv\Scripts\activate.bat
echo   python -m app.test_pipeline
echo.
pause
