# Customer Engagement & Product Utilization Analytics

## Setup (once)
    pip install -r requirements.txt

## 1. Run the analysis (for the research paper)
    python analysis.py
Creates: charts/*.png, findings.txt, at_risk_premium_customers.csv

## 2. Run the dashboard
    streamlit run app.py

## 3. Deploy (free)
Push this folder to a public GitHub repo, go to share.streamlit.io,
choose the repo and app.py, click Deploy. Put the link in your CV.

Files: retention.py (shared logic), analysis.py, app.py, European_Bank.csv
