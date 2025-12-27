Ordner: dashboard/

Start:
  ./run.sh 8501 0.0.0.0 /workspace/runs/run_log.jsonl

Oder manuell:
  pip install -r requirements.txt
  streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501

Tunnel:
  Vast.ai Tunnel-Ziel: http://localhost:8501
