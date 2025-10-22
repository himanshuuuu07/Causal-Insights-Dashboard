import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor

def discover_causal_structure(df, target_metric, method='feature_importance'):
    """Discover causal relationships using feature importance"""
    X = df.drop(columns=[target_metric, 'Date'], errors='ignore')
    y = df[target_metric]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train Random Forest and extract feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    return importance_df, rf, X.columns

def generate_causal_graph(importance_df, top_n=5):
    """Create NetworkX directed graph for visualization"""
    G = nx.DiGraph()

    top_features = importance_df.head(top_n)

    # Add target node (Revenue)
    G.add_node("Revenue", color='red', size=3000)

    # Add top feature nodes with edges to Revenue
    for _, row in top_features.iterrows():
        G.add_node(row['Feature'], color='lightblue', size=1500)
        G.add_edge(row['Feature'], "Revenue", weight=row['Importance'])

    return G

def detect_metric_change(df, target_metric, window=30):
    """Compare recent vs previous period to detect metric changes"""
    recent = df.tail(window)[target_metric].mean()
    previous = df.iloc[-2*window:-window][target_metric].mean()
    pct_change = ((recent - previous) / previous) * 100
    return {
        'recent_avg': recent,
        'previous_avg': previous,
        'pct_change': pct_change,
        'direction': 'increased' if pct_change > 0 else 'decreased'
    }
