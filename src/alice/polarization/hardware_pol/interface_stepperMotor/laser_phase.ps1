# Ensure Execution Policy allows the script to run
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Define paths
$scriptDirectory = (Get-Location).Path  # Get current directory
$venvDir = Join-Path $scriptDirectory ".venv"
$venvActivate = Join-Path $venvDir "Scripts\Activate.ps1"
$pythonScript = Join-Path $scriptDirectory "main.py"
$requirementsFile = Join-Path $scriptDirectory "requirements.txt"

# Check if virtual environment exists, if not create it
if (-Not (Test-Path $venvDir)) {
    Write-Host "Virtual environment not found. Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvDir
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& $venvActivate

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements if requirements.txt exists
if (Test-Path $requirementsFile) {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Green
    pip install -r $requirementsFile
} else {
    Write-Host "Warning: requirements.txt not found. Skipping installation." -ForegroundColor Yellow
}

# Check if main.py exists
if (-Not (Test-Path $pythonScript)) {
    Write-Host "Error: main.py not found in the current directory." -ForegroundColor Red
    exit 1
}

# Run the Python script
Write-Host "Running main.py..." -ForegroundColor Green
python $pythonScript

# Deactivate the virtual environment (optional)
Write-Host "Script execution completed." -ForegroundColor Green
deactivate