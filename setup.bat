@echo off
echo ========================================
echo Flower Classification Model Setup
echo ========================================
echo.

REM Step 1: Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Install Python 3.8+ first.
    pause
    exit /b 1
)
echo Python found!

REM Step 2: Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists.
) else (
    python -m venv venv
    if %errorlevel% equ 0 (
        echo Virtual environment created!
    ) else (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
)

REM Step 3: Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo Activated!

REM Step 4: Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Done!

REM Step 5: Install dependencies
echo.
echo [5/6] Installing dependencies...
echo This may take 5-10 minutes on first run...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo Dependencies installed!
) else (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

REM Step 6: Create directories
echo.
echo [6/6] Creating directories...
if not exist "models" mkdir models
if not exist "uploads" mkdir uploads
if not exist "dataset" mkdir dataset
echo Done!

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Train model:  python train.py
echo 2. Run app:      python app.py
echo 3. Open:         http://127.0.0.1:5000/
echo.
echo See TROUBLESHOOTING.md if you have issues.
echo.
pause
