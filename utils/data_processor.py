import pandas as pd
import numpy as np

def validate_and_clean(df):
    """Clean and validate uploaded data"""
    date_cols = df.select_dtypes(include=['object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def generate_sample_data():
    """Generate demo retail sales dataset with realistic causal relationships"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2025-10-21', freq='D')

    df = pd.DataFrame({
        'Date': dates,
        'Region': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune'], len(dates)),
        'Ad_Spend': np.random.uniform(50000, 200000, len(dates)),
        'Leads': np.random.poisson(150, len(dates)),
        'Conversion_Rate': np.random.uniform(0.05, 0.25, len(dates)),
        'Discount_Percent': np.random.uniform(5, 30, len(dates)),
        'Revenue': 0
    })

    # Create realistic causal relationships
    df['Revenue'] = (
        df['Leads'] * df['Conversion_Rate'] * 5000 -
        df['Discount_Percent'] * 2000 +
        df['Ad_Spend'] * 0.3 +
        np.random.normal(0, 10000, len(dates))
    )

    # Simulate a drop in last 30 days (lower conversion & ad spend)
    df.loc[df['Date'] > '2025-09-21', 'Conversion_Rate'] *= 0.7
    df.loc[df['Date'] > '2025-09-21', 'Ad_Spend'] *= 0.6

    return df
