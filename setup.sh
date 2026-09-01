#!/usr/bin/env bash
# One-command setup for Mac/Linux. Run from the project root: ./setup.sh
set -e

echo "=== 1. Creating virtual environment ==="
python3 -m venv venv

echo "=== 2. Activating it ==="
source venv/bin/activate

echo "=== 3. Installing dependencies ==="
python3 -m pip install --upgrade pip > /dev/null
python3 -m pip install -r requirements.txt

echo "=== 4. Verifying imports ==="
python3 -c "import pandas, sklearn, shap, statsmodels, flask, joblib; print('All dependencies OK')"

echo "=== 5. Running the full batch pipeline ==="
cd src
python3 generate_data.py
python3 features.py
python3 train_eval.py
python3 train_provisioning_model.py
python3 agent.py
python3 investigate.py
python3 build_dashboard.py
cd ..

echo "=== 6. Running tests ==="
cd tests
python3 -m pytest test_pipeline.py -v
cd ..

echo ""
echo "Done. Open docs/dashboard.html to see the results."
echo "For the live real-time demo: cd src, then in one terminal 'python3 live_server.py',"
echo "open http://127.0.0.1:5050/dashboard, and in a second terminal 'python3 simulate_live_stream.py'."
