# Quick Setup Script for Flower Classification Model
# Run this script to set up everything automatically

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Flower Classification Model Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python installation
Write-Host "[0/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create virtual environment
Write-Host "[1/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment created successfully!" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate virtual environment
Write-Host "[2/6] Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
if ($?) {
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not activate via PowerShell. Trying cmd..." -ForegroundColor Yellow
    cmd /c ".\venv\Scripts\activate.bat"
}

# Step 4: Upgrade pip first
Write-Host "[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "pip upgraded successfully!" -ForegroundColor Green

# Step 5: Install dependencies
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes on first run..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    Write-Host "Try running: pip install -r requirements.txt manually" -ForegroundColor Yellow
}

# Step 6: Create necessary directories
Write-Host "[5/6] Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "dataset" | Out-Null
Write-Host "Directories created!" -ForegroundColor Green

# Step 7: Kaggle API setup check
Write-Host "[6/6] Checking Kaggle API configuration..." -ForegroundColor Yellow
$kagglePath = "$env:USERPROFILE\.kaggle\kaggle.json"
if (Test-Path $kagglePath) {
    Write-Host "Kaggle API configured correctly!" -ForegroundColor Green
    Write-Host "Real dataset will be downloaded during training." -ForegroundColor Green
} else {
    Write-Host "WARNING: Kaggle API not configured!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To download the real flower dataset (recommended):" -ForegroundColor Yellow
    Write-Host "1. Create account at kaggle.com" -ForegroundColor Yellow
    Write-Host "2. Go to My Account -> API -> Create New API Token" -ForegroundColor Yellow
    Write-Host "3. Place kaggle.json in: $kagglePath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Training will use synthetic data as fallback (lower accuracy)." -ForegroundColor Yellow
    Write-Host "See DATASET_SETUP.md for detailed instructions." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Train the model:  python train.py" -ForegroundColor White
Write-Host "   (Takes 30-60 mins with real data, 5 mins with synthetic)" -ForegroundColor Gray
Write-Host "2. Run the app:      python app.py" -ForegroundColor White
Write-Host "3. Open browser:     http://127.0.0.1:5000/" -ForegroundColor White
Write-Host ""
Write-Host "For help, see:" -ForegroundColor Cyan
Write-Host "- START_HERE.md : Quick start guide" -ForegroundColor Gray
Write-Host "- README.md : Full documentation" -ForegroundColor Gray
Write-Host "- DATASET_SETUP.md : Kaggle setup guide" -ForegroundColor Gray
Write-Host ""
