import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load dataset
df = pd.read_excel('1. Dataset .xlsx')

# 1. Diagnosis Distribution
fig_diag = px.pie(df, names='Diagnosis', title='Alzheimer\'s Diagnosis Distribution (0: No, 1: Yes)',
             color_discrete_sequence=px.colors.qualitative.Pastel)
html_diag = fig_diag.to_html(full_html=False, include_plotlyjs='cdn')

# 2. Age vs Diagnosis
fig_age = px.histogram(df, x='Age', color='Diagnosis', barmode='overlay',
                   title='Age Distribution by Diagnosis',
                   labels={'Age': 'Age (Years)', 'count': 'Number of Patients'})
html_age = fig_age.to_html(full_html=False, include_plotlyjs='cdn')

# 3. MMSE vs ADL (Core Clinical Metrics)
fig_scatter = px.scatter(df, x='MMSE', y='ADL', color='Diagnosis',
                    title='MMSE vs ADL Scores',
                    hover_data=['Age', 'BMI'],
                    opacity=0.6)
html_scatter = fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')

# 4. Correlation Heatmap (Selected Features)
cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'MMSE', 'ADL', 'Diagnosis']
corr = df[cols].corr()
fig_corr = px.imshow(corr, text_auto=True, title='Correlation Heatmap of Key Features')
html_corr = fig_corr.to_html(full_html=False, include_plotlyjs='cdn')

# 5. Boxplots for Clinical Scores
fig_box = make_subplots(rows=1, cols=2, subplot_titles=("MMSE Distribution", "ADL Distribution"))
fig_box.add_trace(go.Box(y=df['MMSE'], name='MMSE', boxpoints='all'), row=1, col=1)
fig_box.add_trace(go.Box(y=df['ADL'], name='ADL', boxpoints='all'), row=1, col=2)
fig_box.update_layout(title_text="Clinical Score Distributions")
html_box = fig_box.to_html(full_html=False, include_plotlyjs='cdn')

# 6. Ethnicity and Gender Analysis
fig_eth = px.bar(df.groupby(['Ethnicity', 'Diagnosis']).size().reset_index(name='count'),
                 x='Ethnicity', y='count', color='Diagnosis', barmode='group',
                 title='Diagnosis by Ethnicity')
html_eth = fig_eth.to_html(full_html=False, include_plotlyjs='cdn')

# Combine everything into a single HTML report
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Alzheimer's Clinical Data EDA</title>
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f8fafc; padding: 20px; color: #1e293b; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ text-align: center; color: #0f172a; margin-bottom: 40px; }}
        .viz-section {{ margin-bottom: 60px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .description {{ margin-top: 15px; color: #64748b; font-size: 14px; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Alzheimer's Disease Clinical Dataset Analysis</h1>

        <div class="viz-section">
            {html_diag}
            <p class="description">This chart shows the balance of the dataset between diagnosed patients and control groups.</p>
        </div>

        <div class="viz-section">
            {html_age}
            <p class="description">Age distribution helps identify the primary demographic affected by the disease in this study.</p>
        </div>

        <div class="viz-section">
            {html_scatter}
            <p class="description">The relationship between MMSE (cognitive) and ADL (functional) scores is a primary indicator of disease progression.</p>
        </div>

        <div class="viz-section">
            {html_corr}
            <p class="description">Correlation heatmap showing which lifestyle and clinical factors correlate most strongly with a diagnosis.</p>
        </div>

        <div class="viz-section">
            {html_box}
            <p class="description">Boxplots showing the range and outliers for the most critical clinical metrics.</p>
        </div>

        <div class="viz-section">
            {html_eth}
            <p class="description">Demographic breakdown of the dataset across different ethnicities.</p>
        </div>
    </div>
</body>
</html>
"""

with open('eda_report.html', 'w') as f:
    f.write(html_template)

print("EDA report generated: eda_report.html")
