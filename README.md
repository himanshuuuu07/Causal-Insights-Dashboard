# 🔍 Causal Insights Dashboard

Automatically discover why your business metrics changed — not just what happened.

## Features
- Upload CSV or use included demo dataset
- Select target metric, auto causal discovery
- Visual causal graph + AI summary
- What-if scenario queries

## Quick Start
1. pip install -r requirements.txt
2. streamlit run app.py
3. Open http://localhost:8501

## Structure
- app.py
- utils/data_processor.py
- utils/causal_analysis.py
- utils/ai_summary.py
- data/sample_sales_data.csv
- .streamlit/config.toml