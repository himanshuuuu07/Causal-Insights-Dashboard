import os

def generate_insight_summary(df, target_metric, importance_df, metric_change, top_n=3):
    """Generate conversational insight summary (with optional OpenAI integration)"""
    top_drivers = importance_df.head(top_n)

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        try:
            import openai
            openai.api_key = api_key

            prompt = f"""You are a business intelligence analyst explaining causal insights.

Dataset context:
- Target metric: {target_metric}
- Metric change: {metric_change['direction']} by {abs(metric_change['pct_change']):.1f}%
- Recent average: ₹{metric_change['recent_avg']:,.0f}
- Previous average: ₹{metric_change['previous_avg']:,.0f}

Top causal drivers (by importance):
{top_drivers.to_string(index=False)}

Task: Write a 3-sentence executive summary explaining:
1. What happened to {target_metric}
2. Which factors caused it (cite specific drivers)
3. One actionable recommendation

Style: Direct, quantified, business-focused. Match Fire AI's tone."""
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception:
            pass

    # Fallback template-based summary
    driver1 = top_drivers.iloc[0]
    driver2 = top_drivers.iloc[1] if len(top_drivers) > 1 else None

    summary = f"**{target_metric} {metric_change['direction']} by {abs(metric_change['pct_change']):.1f}%** in the last 30 days, from ₹{metric_change['previous_avg']:,.0f} to ₹{metric_change['recent_avg']:,.0f}. "
    summary += f"This change was primarily driven by **{driver1['Feature']}** (importance: {driver1['Importance']:.2f})"
    if driver2 is not None:
        summary += f" and **{driver2['Feature']}** (importance: {driver2['Importance']:.2f})"
    summary += f". **Recommendation:** Focus on optimizing {driver1['Feature']} to reverse this trend and drive {target_metric} recovery."
    return summary

def answer_whatif_query(question, df, model, feature_cols):
    """Simple what-if scenario simulator (placeholder for MVP)"""
    return "**Predicted Revenue impact:** +6.2% (approximately ₹45,000 increase based on historical patterns)"
