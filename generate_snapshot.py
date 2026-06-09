import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from datetime import datetime
import json

# Define the custom encoder globally
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_full_snapshot():
    # Load dataset
    df = pd.read_excel('1. Dataset .xlsx')

    # Preprocessing
    numeric_cols_to_fix = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'Diagnosis']
    for col in numeric_cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=['Diagnosis', 'MMSE', 'ADL']).copy()
    df_clean['Diagnosis_Label'] = df_clean['Diagnosis'].map({0: 'Healthy', 1: 'Alzheimer\'s'}).fillna('Unknown')
    df_clean['Gender_Label'] = df_clean['Gender'].map({0: 'Male', 1: 'Female'}).fillna('Unknown')

    # 1. Population Averages
    averages = {
        "MMSE": round(df_clean['MMSE'].mean(), 2),
        "ADL": round(df_clean['ADL'].mean(), 2),
        "FunctionalAssessment": round(df_clean['FunctionalAssessment'].mean(), 2),
        "BMI": round(df_clean['BMI'].mean(), 2),
        "Age": round(df_clean['Age'].mean(), 2)
    }

    # 3. Interactive EDA Charts (Plotly)
    # Demographic & Diagnosis
    fig_sunburst = px.sunburst(df_clean, path=['Gender_Label', 'Diagnosis_Label'],
                              color='Diagnosis_Label',
                              color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_sunburst.update_layout(title="Demographic Distribution by Diagnosis", font_family="Inter")

    # Correlation Heatmap
    corr_cols = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL', 'SystolicBP', 'CholesterolTotal']
    valid_corr_cols = [col for col in corr_cols if df_clean[col].nunique() > 1]
    corr = df_clean[valid_corr_cols].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values.tolist(),
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=corr.values.round(2).astype(str).tolist(),
        texttemplate="%{text}"
    ))
    fig_heatmap.update_layout(title="Clinical Metric Correlation Matrix", font_family="Inter")

    # Age Violin
    fig_violin = px.violin(df_clean, y="Age", x="Diagnosis_Label", color="Diagnosis_Label",
                          box=True, points=None,
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_violin.update_layout(title="Age Distribution across Diagnosis Groups", font_family="Inter")

    # Diagnosis Pie
    fig_pie = px.pie(df_clean, names='Diagnosis_Label', color='Diagnosis_Label',
                    color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_pie.update_layout(title="Population Diagnosis Prevalence", font_family="Inter")

    # MMSE vs ADL Scatter - USING GRAPH OBJECTS TO AVOID BINARY DATA
    fig_scatter = go.Figure()
    for label, color in [('Healthy', '#10b981'), ("Alzheimer's", '#ef4444')]:
        mask = df_clean['Diagnosis_Label'] == label
        fig_scatter.add_trace(go.Scatter(
            x=df_clean[mask]['MMSE'].tolist(),
            y=df_clean[mask]['ADL'].tolist(),
            mode='markers',
            name=label,
            marker=dict(color=color, opacity=0.6)
        ))
    fig_scatter.update_layout(title="Cognitive vs Functional Performance",
                             xaxis_title="MMSE Score (Cognitive)",
                             yaxis_title="ADL Score (Functional)",
                             font_family="Inter")

    # Lifestyle Factors Bar
    lifestyle_cols = ['Smoking', 'AlcoholConsumption', 'PhysicalActivity']
    lifestyle_df = df_clean.groupby('Diagnosis_Label')[lifestyle_cols].mean().reset_index()
    lifestyle_melted = lifestyle_df.melt(id_vars='Diagnosis_Label', var_name='Factor', value_name='Prevalence')
    fig_lifestyle = px.bar(lifestyle_melted, x='Factor', y='Prevalence', color='Diagnosis_Label', barmode='group',
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_lifestyle.update_layout(title="Prevalence of Lifestyle Risk Factors", font_family="Inter")

    # Education Level
    edu_map = {0: 'None', 1: 'High School', 2: 'Bachelor', 3: 'Higher'}
    df_clean['Education_Label'] = df_clean['EducationLevel'].map(edu_map).fillna('Unknown')
    fig_edu = px.histogram(df_clean, x="Education_Label", color="Diagnosis_Label", barmode='group',
                          category_orders={"Education_Label": ["None", "High School", "Bachelor", "Higher"]},
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_edu.update_layout(title="Education Level vs Diagnosis", font_family="Inter")

    # Cardiovascular Distribution
    fig_bp = px.box(df_clean, x="Diagnosis_Label", y="SystolicBP", color="Diagnosis_Label",
                   color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_bp.update_layout(title="Systolic Blood Pressure Distribution", font_family="Inter")

    # Template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Alzheimer's Clinical Insights - Data Snapshot</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/inter-ui@3.19.3/inter.min.css">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #10b981;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #0f172a;
            --border: #e2e8f0;
            --radius: 16px;
        }
        body {
            font-family: "Inter", sans-serif;
            background-color: #f1f5f9;
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.5;
        }
        .navbar {
            background: #ffffff;
            color: var(--primary);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .nav-links button {
            background: #f1f5f9;
            border: none;
            color: var(--secondary);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-left: 10px;
            transition: all 0.2s;
        }
        .nav-links button.active {
            background: var(--primary);
            color: white;
        }
        .container {
            max-width: 1280px;
            margin: 2rem auto;
            padding: 0 20px;
        }
        .section { display: none; }
        .section.active { display: block; }

        .card {
            background: var(--card);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 30px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border);
        }
        .metric-card label {
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--secondary);
            margin-bottom: 8px;
        }
        .metric-card span {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary);
        }

        .chart-header {
            margin-bottom: 15px;
            border-left: 4px solid var(--primary);
            padding-left: 15px;
        }
        .chart-header h3 { margin: 0; font-size: 1.25rem; }
        .clinical-note {
            margin-top: 15px;
            background: #eff6ff;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #1e40af;
            border-left: 4px solid #3b82f6;
        }
        .clinical-note b { color: #1e3a8a; }

        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 30px;
        }
        .plotly-chart { height: 450px; width: 100%; min-height: 450px; }

        @media (max-width: 768px) {
            .viz-grid { grid-template-columns: 1fr; }
            .plotly-chart { height: 350px; min-height: 350px; }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div style="font-size: 1.25rem; font-weight: 800;">🧠 Clinical Data Explorer</div>
        <div class="nav-links">
            <button onclick="showSection('summary')" id="btn-summary" class="active">Overview</button>
            <button onclick="showSection('cognitive')" id="btn-cognitive">Cognitive & Lifestyle</button>
            <button onclick="showSection('biomedical')" id="btn-biomedical">Biomedical Factors</button>
        </div>
    </nav>

    <div class="container">
        <div id="section-summary" class="section active">
            <div class="metric-grid">
                <div class="metric-card"><label>Total Samples</label><span>{{ total_count }}</span></div>
                <div class="metric-card"><label>Avg Age</label><span>{{ averages.Age }}</span></div>
                <div class="metric-card"><label>Avg MMSE</label><span>{{ averages.MMSE }}</span></div>
                <div class="metric-card"><label>Avg ADL</label><span>{{ averages.ADL }}</span></div>
            </div>
            <div class="viz-grid">
                <div class="card">
                    <div class="chart-header"><h3>Diagnosis Prevalence</h3></div>
                    <div id="pieChart" class="plotly-chart"></div>
                    <div class="clinical-note"><b>Note:</b> Baseline distribution of the cohort.</div>
                </div>
                <div class="card">
                    <div class="chart-header"><h3>Demographic Distribution</h3></div>
                    <div id="sunburstChart" class="plotly-chart"></div>
                    <div class="clinical-note"><b>Note:</b> Intersection of Gender and Diagnosis.</div>
                </div>
            </div>
            <div class="card">
                <div class="chart-header"><h3>Age Distribution Across Diagnosis</h3></div>
                <div id="violinChart" class="plotly-chart"></div>
                <div class="clinical-note"><b>Note:</b> Age remains a primary risk factor.</div>
            </div>
        </div>

        <div id="section-cognitive" class="section">
            <div class="card">
                <div class="chart-header"><h3>Cognitive (MMSE) vs Functional (ADL) Correlation</h3></div>
                <div id="scatterChart" class="plotly-chart"></div>
                <div class="clinical-note"><b>Clinical Significance:</b> relationship between cognitive decline and daily activities.</div>
            </div>
            <div class="viz-grid">
                <div class="card">
                    <div class="chart-header"><h3>Lifestyle Risk Factors</h3></div>
                    <div id="lifestyleChart" class="plotly-chart"></div>
                </div>
                <div class="card">
                    <div class="chart-header"><h3>Education as a Protective Factor</h3></div>
                    <div id="eduChart" class="plotly-chart"></div>
                </div>
            </div>
        </div>

        <div id="section-biomedical" class="section">
            <div class="card">
                <div class="chart-header"><h3>Clinical Metric Correlation Matrix</h3></div>
                <div id="heatmapChart" class="plotly-chart" style="height: 600px;"></div>
                <div class="clinical-note"><b>Clinical Significance:</b> relationships between biomarkers and clinical outcomes.</div>
            </div>
            <div class="card">
                <div class="chart-header"><h3>Vascular Health: Systolic BP</h3></div>
                <div id="bpChart" class="plotly-chart"></div>
            </div>
        </div>
    </div>

    <script>
        const chartData = {
            pie: {{ fig_pie_json | safe }},
            sunburst: {{ fig_sunburst_json | safe }},
            violin: {{ fig_violin_json | safe }},
            scatter: {{ fig_scatter_json | safe }},
            lifestyle: {{ fig_lifestyle_json | safe }},
            edu: {{ fig_edu_json | safe }},
            heatmap: {{ fig_heatmap_json | safe }},
            bp: {{ fig_bp_json | safe }}
        };

        const config = { responsive: true, displayModeBar: true, displaylogo: false };

        function initCharts() {
            Plotly.newPlot('pieChart', chartData.pie.data, chartData.pie.layout, config);
            Plotly.newPlot('sunburstChart', chartData.sunburst.data, chartData.sunburst.layout, config);
            Plotly.newPlot('violinChart', chartData.violin.data, chartData.violin.layout, config);
            Plotly.newPlot('scatterChart', chartData.scatter.data, chartData.scatter.layout, config);
            Plotly.newPlot('lifestyleChart', chartData.lifestyle.data, chartData.lifestyle.layout, config);
            Plotly.newPlot('eduChart', chartData.edu.data, chartData.edu.layout, config);
            Plotly.newPlot('heatmapChart', chartData.heatmap.data, chartData.heatmap.layout, config);
            Plotly.newPlot('bpChart', chartData.bp.data, chartData.bp.layout, config);
            window.dispatchEvent(new Event('resize'));
        }

        window.onload = initCharts;

        function showSection(id) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
            document.getElementById('section-' + id).classList.add('active');
            document.getElementById('btn-' + id).classList.add('active');
            setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 100);
        }
    </script>
</body>
</html>
    """

    # Prepare JSONs with NpEncoder
    t = Template(html_template)
    def to_json_clean(fig):
        return json.dumps(fig.to_dict(), cls=NpEncoder)

    html_output = t.render(
        averages=averages,
        total_count=len(df_clean),
        fig_pie_json=to_json_clean(fig_pie),
        fig_sunburst_json=to_json_clean(fig_sunburst),
        fig_heatmap_json=to_json_clean(fig_heatmap),
        fig_violin_json=to_json_clean(fig_violin),
        fig_scatter_json=to_json_clean(fig_scatter),
        fig_lifestyle_json=to_json_clean(fig_lifestyle),
        fig_edu_json=to_json_clean(fig_edu),
        fig_bp_json=to_json_clean(fig_bp)
    )

    with open('site_snapshot.html', 'w') as f:
        f.write(html_output)

    print("Corrected medical snapshot generated: site_snapshot.html")

if __name__ == "__main__":
    generate_full_snapshot()
