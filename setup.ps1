# One-command setup for Windows PowerShell.
# Run from the project root: .\setup.ps1
# Creates a venv, installs dependencies, and runs the full batch pipeline
# end to end, printing every result along the way.

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Creating virtual environment ===" -ForegroundColor Cyan
python3 -m venv venv

Write-Host "=== 2. Activating it ===" -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host "=== 3. Installing dependencies ===" -ForegroundColor Cyan
python3 -m pip install --upgrade pip | Out-Null
python3 -m pip install -r requirements.txt

Write-Host "=== 4. Verifying imports ===" -ForegroundColor Cyan
python3 -c "import pandas, sklearn, shap, statsmodels, flask, joblib; print('All dependencies OK')"

Write-Host "=== 5. Running the full batch pipeline ===" -ForegroundColor Cyan
Set-Location src
python3 generate_data.py
python3 features.py
python3 train_eval.py
python3 train_provisioning_model.py
python3 agent.py
python3 investigate.py
python3 build_dashboard.py
Set-Location ..

Write-Host "=== 6. Running tests ===" -ForegroundColor Cyan
Set-Location tests
python3 -m pytest test_pipeline.py -v
Set-Location ..

Write-Host ""
Write-Host "Done. Open docs\dashboard.html to see the results." -ForegroundColor Green
Write-Host "For the live real-time demo: cd src, then in one terminal 'python3 live_server.py'," -ForegroundColor Green
Write-Host "open http://127.0.0.1:5050/dashboard, and in a second terminal 'python3 simulate_live_stream.py'." -ForegroundColor Green
