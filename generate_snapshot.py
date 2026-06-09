import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from datetime import datetime
import json

def generate_full_snapshot():
    # Load dataset
    df = pd.read_excel('1. Dataset .xlsx')

    # Preprocessing
    # Ensure numeric types for clinical scores
    for col in ['MMSE', 'ADL', 'FunctionalAssessment', 'Age', 'BMI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=['Gender', 'Diagnosis', 'MMSE', 'ADL'])
    df_clean['Diagnosis_Label'] = df_clean['Diagnosis'].map({0: 'Healthy', 1: 'Alzheimer\'s'})
    df_clean['Gender_Label'] = df_clean['Gender'].map({0: 'Male', 1: 'Female'})

    # 1. Population Averages
    averages = {
        "MMSE": round(df_clean['MMSE'].mean(), 2),
        "ADL": round(df_clean['ADL'].mean(), 2),
        "FunctionalAssessment": round(df_clean['FunctionalAssessment'].mean(), 2),
        "BMI": round(df_clean['BMI'].mean(), 2),
        "Age": round(df_clean['Age'].mean(), 2)
    }

    # 2. Distribution Stats (for Chart.js)
    mmse_hist, mmse_edges = np.histogram(df_clean['MMSE'].dropna(), bins=20)
    adl_hist, adl_edges = np.histogram(df_clean['ADL'].dropna(), bins=20)
    stats = {
        "mmse_bins": [round(b, 1) for b in mmse_edges[:-1]],
        "mmse_counts": mmse_hist.tolist(),
        "adl_bins": [round(b, 1) for b in adl_edges[:-1]],
        "adl_counts": adl_hist.tolist()
    }

    # 3. Interactive EDA Charts (Plotly)
    # Demographic & Diagnosis
    fig_sunburst = px.sunburst(df_clean, path=['Gender_Label', 'Diagnosis_Label'],
                              color='Diagnosis_Label',
                              color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                              title="Demographic Distribution by Diagnosis")

    # Correlation Heatmap
    numeric_cols = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL', 'Diagnosis', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal']
    corr = df_clean[numeric_cols].corr()
    fig_heatmap = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                           title="Clinical Metric Correlations")

    # Age Violin
    fig_violin = px.violin(df_clean, y="Age", x="Diagnosis_Label", color="Diagnosis_Label",
                          box=True, points="all", color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                          title="Age Distribution across Diagnosis Groups")

    # Diagnosis Pie
    fig_pie = px.pie(df_clean, names='Diagnosis_Label', color='Diagnosis_Label',
                    color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                    title="Population Diagnosis Prevalence")

    # MMSE vs ADL Scatter (The one that appeared empty in screenshot)
    fig_scatter = px.scatter(df_clean, x="MMSE", y="ADL", color="Diagnosis_Label",
                            marginal_x="histogram", marginal_y="box",
                            color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                            title="Cognitive vs Functional Performance",
                            labels={'MMSE': 'MMSE Score (Cognitive)', 'ADL': 'ADL Score (Functional)'})

    # Lifestyle Factors Bar
    lifestyle_cols = ['Smoking', 'AlcoholConsumption', 'PhysicalActivity']
    lifestyle_df = df_clean.groupby('Diagnosis_Label')[lifestyle_cols].mean().reset_index()
    lifestyle_melted = lifestyle_df.melt(id_vars='Diagnosis_Label', var_name='Factor', value_name='Prevalence')
    fig_lifestyle = px.bar(lifestyle_melted, x='Factor', y='Prevalence', color='Diagnosis_Label', barmode='group',
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                          title="Prevalence of Lifestyle Risk Factors")

    # Education Level
    edu_map = {0: 'None', 1: 'High School', 2: 'Bachelor', 3: 'Higher'}
    df_clean['Education_Label'] = df_clean['EducationLevel'].map(edu_map)
    fig_edu = px.histogram(df_clean, x="Education_Label", color="Diagnosis_Label", barmode='group',
                          category_orders={"Education_Label": ["None", "High School", "Bachelor", "Higher"]},
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                          title="Education Level vs Diagnosis")

    # Cardiovascular Distribution
    fig_bp = px.box(df_clean, x="Diagnosis_Label", y="SystolicBP", color="Diagnosis_Label",
                   color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'},
                   title="Systolic Blood Pressure Distribution")

    # Template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Alzheimer's Clinical Insights - Data Snapshot</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/inter-ui@3.19.3/inter.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
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
        .section { display: none; opacity: 0; }
        .section.active { display: block; opacity: 1; }

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
        .metric-card value {
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
        .plotly-chart { height: 450px; }

        @media (max-width: 768px) {
            .viz-grid { grid-template-columns: 1fr; }
            .plotly-chart { height: 350px; }
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
        <!-- Overview Section -->
        <div id="section-summary" class="section active">
            <div class="metric-grid">
                <div class="metric-card"><label>Total Samples</label><value>{{ total_count }}</value></div>
                <div class="metric-card"><label>Avg Age</label><value>{{ averages.Age }}</value></div>
                <div class="metric-card"><label>Avg MMSE</label><value>{{ averages.MMSE }}</value></div>
                <div class="metric-card"><label>Avg ADL</label><value>{{ averages.ADL }}</value></div>
            </div>

            <div class="viz-grid">
                <div class="card">
                    <div class="chart-header">
                        <h3>Diagnosis Prevalence</h3>
                    </div>
                    <div id="pieChart" class="plotly-chart"></div>
                    <div class="clinical-note">
                        <b>Clinical Significance:</b> Provides the baseline distribution of the cohort. High prevalence of Alzheimer's in this dataset allows for robust comparative analysis of risk factors.
                    </div>
                </div>
                <div class="card">
                    <div class="chart-header">
                        <h3>Demographic Distribution</h3>
                    </div>
                    <div id="sunburstChart" class="plotly-chart"></div>
                    <div class="clinical-note">
                        <b>Clinical Significance:</b> Visualizes the intersection of Gender and Diagnosis. Gender is often studied as a biological variable in disease progression and prevalence.
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="chart-header">
                    <h3>Age Distribution Across Diagnosis</h3>
                </div>
                <div id="violinChart" class="plotly-chart"></div>
                <div class="clinical-note">
                    <b>Clinical Significance:</b> Age is the primary risk factor for Alzheimer's. This violin plot shows how the diagnosis concentration shifts across different age cohorts, highlighting the typical late-onset window.
                </div>
            </div>
        </div>

        <!-- Cognitive & Lifestyle Section -->
        <div id="section-cognitive" class="section">
            <div class="card">
                <div class="chart-header">
                    <h3>Cognitive (MMSE) vs Functional (ADL) Correlation</h3>
                </div>
                <div id="scatterChart" class="plotly-chart"></div>
                <div class="clinical-note">
                    <b>Clinical Significance:</b> Explores the relationship between cognitive decline (MMSE) and the ability to perform daily activities (ADL). A strong correlation indicates that cognitive loss directly impacts patient independence.
                </div>
            </div>

            <div class="viz-grid">
                <div class="card">
                    <div class="chart-header">
                        <h3>Lifestyle Risk Factors</h3>
                    </div>
                    <div id="lifestyleChart" class="plotly-chart"></div>
                    <div class="clinical-note">
                        <b>Clinical Significance:</b> Modifiable risk factors. Shows the mean prevalence of smoking, alcohol, and physical activity, which are critical targets for preventative medicine.
                    </div>
                </div>
                <div class="card">
                    <div class="chart-header">
                        <h3>Education as a Protective Factor</h3>
                    </div>
                    <div id="eduChart" class="plotly-chart"></div>
                    <div class="clinical-note">
                        <b>Clinical Significance:</b> Higher education levels are often associated with "cognitive reserve," which may delay the onset of clinical symptoms despite underlying pathology.
                    </div>
                </div>
            </div>
        </div>

        <!-- Biomedical Section -->
        <div id="section-biomedical" class="section">
            <div class="card">
                <div class="chart-header">
                    <h3>Clinical Metric Correlation Matrix</h3>
                </div>
                <div id="heatmapChart" class="plotly-chart" style="height: 600px;"></div>
                <div class="clinical-note">
                    <b>Clinical Significance:</b> Identifies hidden relationships between biomarkers (BMI, BP, Cholesterol) and clinical outcomes (Diagnosis, MMSE). Red indicates positive correlation; blue indicates negative.
                </div>
            </div>

            <div class="card">
                <div class="chart-header">
                    <h3>Vascular Health: Systolic BP</h3>
                </div>
                <div id="bpChart" class="plotly-chart"></div>
                <div class="clinical-note">
                    <b>Clinical Significance:</b> Hypertension is a major vascular risk factor for dementia. This box plot compares blood pressure levels between healthy individuals and those with a diagnosis.
                </div>
            </div>
        </div>
    </div>

    <script>
        function showSection(id) {
            document.querySelectorAll('.section').forEach(s => {
                s.classList.remove('active');
                s.style.display = 'none';
            });
            document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));

            const activeSec = document.getElementById('section-' + id);
            activeSec.style.display = 'block';
            setTimeout(() => {
                activeSec.classList.add('active');
                window.dispatchEvent(new Event('resize'));
            }, 50);

            document.getElementById('btn-' + id).classList.add('active');
        }

        const config = { responsive: true, displayModeBar: true, displaylogo: false };
        const layout = { margin: { t: 40, b: 40, l: 60, r: 20 }, font: { family: 'Inter' } };

        Plotly.newPlot('pieChart', {{ fig_pie_json | safe }}.data, {{ fig_pie_json | safe }}.layout, config);
        Plotly.newPlot('sunburstChart', {{ fig_sunburst_json | safe }}.data, {{ fig_sunburst_json | safe }}.layout, config);
        Plotly.newPlot('violinChart', {{ fig_violin_json | safe }}.data, {{ fig_violin_json | safe }}.layout, config);
        Plotly.newPlot('scatterChart', {{ fig_scatter_json | safe }}.data, {{ fig_scatter_json | safe }}.layout, config);
        Plotly.newPlot('lifestyleChart', {{ fig_lifestyle_json | safe }}.data, {{ fig_lifestyle_json | safe }}.layout, config);
        Plotly.newPlot('eduChart', {{ fig_edu_json | safe }}.data, {{ fig_edu_json | safe }}.layout, config);
        Plotly.newPlot('heatmapChart', {{ fig_heatmap_json | safe }}.data, {{ fig_heatmap_json | safe }}.layout, config);
        Plotly.newPlot('bpChart', {{ fig_bp_json | safe }}.data, {{ fig_bp_json | safe }}.layout, config);
    </script>
</body>
</html>
    """

    t = Template(html_template)
    html_output = t.render(
        patient=averages,
        averages=averages,
        stats=stats,
        total_count=len(df_clean),
        date=datetime.now().strftime("%Y-%m-%d"),
        fig_pie_json=fig_pie.to_json(),
        fig_sunburst_json=fig_sunburst.to_json(),
        fig_heatmap_json=fig_heatmap.to_json(),
        fig_violin_json=fig_violin.to_json(),
        fig_scatter_json=fig_scatter.to_json(),
        fig_lifestyle_json=fig_lifestyle.to_json(),
        fig_edu_json=fig_edu.to_json(),
        fig_bp_json=fig_bp.to_json()
    )

    with open('site_snapshot.html', 'w') as f:
        f.write(html_output)

    print("Enhanced professional medical snapshot generated: site_snapshot.html")

if __name__ == "__main__":
    generate_full_snapshot()
