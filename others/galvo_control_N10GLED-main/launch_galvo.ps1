# Ensure Execution Policy allows the script to run
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Define paths
$scriptDirectory = (Get-Location).Path  # Get current directory
$venvPath = Join-Path $scriptDirectory ".venv\Scripts\Activate.ps1"
$pythonScriptPath = Join-Path $scriptDirectory "main.py"

# Check if .venv exists
if (-Not (Test-Path $venvPath)) {
    Write-Host "Error: Virtual environment not found at .venv. Please create it first." -ForegroundColor Red
    exit 1
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& $venvPath

# Check if the main.py script exists
if (-Not (Test-Path $pythonScriptPath)) {
    Write-Host "Error: main.py not found in the current directory." -ForegroundColor Red
    exit 1
}

# Run the Python script
Write-Host "Running main.py..." -ForegroundColor Green
python $pythonScriptPath

# Deactivate virtual environment (optional, if it's set up in your venv)
Write-Host "Script execution completed." -ForegroundColor Green
