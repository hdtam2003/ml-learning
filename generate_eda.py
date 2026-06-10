import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_report():
    # Load dataset
    df = pd.read_excel('1. Dataset .xlsx')

    # Preprocessing for visualization
    df = df.dropna(subset=['Gender', 'Diagnosis'])
    df['Diagnosis_Label'] = df['Diagnosis'].map({0: 'Healthy', 1: 'Alzheimer\'s'})
    df['Gender_Label'] = df['Gender'].map({0: 'Male', 1: 'Female'})

    # 1. Diagnosis Distribution by Gender
    fig1 = px.sunburst(df, path=['Gender_Label', 'Diagnosis_Label'],
                      title='Population Breakdown by Gender and Diagnosis',
                      color='Diagnosis_Label',
                      color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    # 2. Cognitive Metrics Correlation
    numeric_cols = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL']
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                    title="Correlation Heatmap: Clinical Metrics",
                    color_continuous_scale='RdBu_r')

    # 3. Age & BMI distribution by Diagnosis
    fig3 = px.violin(df, y="Age", x="Diagnosis_Label", color="Diagnosis_Label",
                    box=True, points="all", title="Age Distribution across Diagnosis Groups",
                    color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    # 4. ADL vs MMSE Scatter
    fig4 = px.scatter(df, x="MMSE", y="ADL", color="Diagnosis_Label",
                     marginal_x="histogram", marginal_y="rug",
                     title="ADL vs MMSE Score Relationship",
                     labels={'MMSE': 'MMSE Score (Cognitive)', 'ADL': 'ADL Score (Functional)'},
                     color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    # Combine into a single HTML report with a clean layout
    with open('eda_report.html', 'w') as f:
        f.write("<html><head>")
        f.write("<meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        f.write("<title>Alzheimer's Clinical EDA</title>")
        f.write("<style>")
        f.write("body{font-family:sans-serif; background:#f8fafc; padding:20px; margin:0;}")
        f.write(".grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:20px;}")
        f.write(".chart-container{background:white; border-radius:8px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,0.1); min-height:450px;}")
        f.write("@media (max-width: 600px) { .grid{grid-template-columns: 1fr;} .chart-container{min-height:350px; padding:10px;} }")
        f.write("</style>")
        f.write("</head><body>")
        f.write("<h1>Advanced Clinical Data Exploration</h1>")
        f.write("<div class='grid'>")

        for i, fig in enumerate([fig1, fig2, fig3, fig4]):
            f.write(f"<div class='chart-container'>{fig.to_html(full_html=False, include_plotlyjs='cdn' if i==0 else False)}</div>")

        f.write("</div></body></html>")

    print("EDA Report generated: eda_report.html")

if __name__ == "__main__":
    generate_report()
