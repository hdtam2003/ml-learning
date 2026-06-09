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
    df_clean = df.dropna(subset=['Gender', 'Diagnosis'])
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
    fig_sunburst = px.sunburst(df_clean, path=['Gender_Label', 'Diagnosis_Label'],
                              color='Diagnosis_Label',
                              color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    numeric_cols = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL', 'Diagnosis']
    corr = df_clean[numeric_cols].corr()
    fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')

    fig_violin = px.violin(df_clean, y="Age", x="Diagnosis_Label", color="Diagnosis_Label",
                          box=True, points="all", color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    fig_pie = px.pie(df_clean, names='Diagnosis_Label', color='Diagnosis_Label',
                    color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    fig_scatter = px.scatter(df_clean, x="MMSE", y="ADL", color="Diagnosis_Label",
                            marginal_x="histogram", marginal_y="rug",
                            color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444'})

    # Template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alzheimer's Clinical Portal - Full Snapshot</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/inter-ui@3.19.3/inter.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --bg: #f1f5f9;
            --card: rgba(255, 255, 255, 0.95);
            --text: #1e293b;
            --border: #e2e8f0;
            --radius: 16px;
        }
        body {
            font-family: "Inter", sans-serif;
            background-color: #cbd5e1;
            color: var(--text);
            margin: 0;
            padding: 0;
        }
        .bg-overlay {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: url('background.jpg');
            background-size: cover;
            background-position: center;
            opacity: 0.15;
            z-index: -1;
        }
        .navbar {
            background: var(--primary);
            color: white;
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav-links button {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin-left: 10px;
        }
        .nav-links button.active {
            background: white;
            color: var(--primary);
        }
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            background: var(--card);
            border-radius: var(--radius);
            padding: 40px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .section { display: none; }
        .section.active { display: block; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--border); padding-bottom: 20px; margin-bottom: 30px; }
        .patient-info { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .info-card { background: rgba(248, 250, 252, 0.8); padding: 20px; border-radius: 12px; border: 1px solid var(--border); }
        .info-card label { display: block; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 5px; }
        .info-card span { font-size: 20px; font-weight: 800; color: #0f172a; }
        .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-top: 40px; }
        .viz-card { background: rgba(248, 250, 252, 0.5); padding: 25px; border-radius: 12px; border: 1px solid var(--border); }
        .chart-container { height: 400px; position: relative; }
        .plotly-chart { width: 100%; height: 100%; }
        h2, h3 { color: #0f172a; }

        @media (max-width: 768px) {
            .navbar { padding: 1rem; flex-direction: column; gap: 10px; }
            .container { margin: 1rem; padding: 20px; border-radius: 8px; }
            .patient-info { grid-template-columns: 1fr 1fr; }
            .viz-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 480px) {
            .patient-info { grid-template-columns: 1fr; }
            .header h1 { font-size: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="bg-overlay"></div>
    <nav class="navbar">
        <div style="font-size: 1.25rem; font-weight: 800;">Clinical Dashboard</div>
        <div class="nav-links">
            <button onclick="showSection('report')" id="btn-report" class="active">Patient Report</button>
            <button onclick="showSection('eda')" id="btn-eda">Population EDA</button>
        </div>
    </nav>

    <div class="container">
        <!-- Patient Report Section -->
        <div id="section-report" class="section active">
            <div class="header">
                <h1>Representative Patient Report</h1>
                <div class="date">{{ date }}</div>
            </div>

            <div class="patient-info">
                <div class="info-card"><label>Age</label><span>{{ patient.Age }}</span></div>
                <div class="info-card"><label>BMI</label><span>{{ patient.BMI }}</span></div>
                <div class="info-card"><label>MMSE</label><span>{{ patient.MMSE }} / 30</span></div>
                <div class="info-card"><label>ADL</label><span>{{ patient.ADL }} / 10</span></div>
                <div class="info-card"><label>Functional Assessment</label><span>{{ patient.FunctionalAssessment }}</span></div>
                <div class="info-card"><label>Total Records</label><span>{{ total_count }}</span></div>
            </div>

            <div class="viz-grid">
                <div class="viz-card">
                    <h3>Clinical Radar</h3>
                    <div class="chart-container"><canvas id="radarChart"></canvas></div>
                </div>
                <div class="viz-card">
                    <h3>MMSE Score Density</h3>
                    <div class="chart-container"><canvas id="mmseDistChart"></canvas></div>
                </div>
            </div>
        </div>

        <!-- EDA Section -->
        <div id="section-eda" class="section">
            <div class="header">
                <h1>Population Analysis Insights</h1>
            </div>

            <div class="viz-grid">
                <div class="viz-card">
                    <h3>Diagnosis Breakdown</h3>
                    <div class="chart-container" id="pieChart"></div>
                </div>
                <div class="viz-card">
                    <h3>Demographic Hierarchy</h3>
                    <div class="chart-container" id="sunburstChart"></div>
                </div>
            </div>

            <div class="viz-card" style="margin-top: 30px;">
                <h3>Correlation Heatmap</h3>
                <div style="height: 500px;" id="heatmapChart"></div>
            </div>

            <div class="viz-grid">
                <div class="viz-card">
                    <h3>Age vs Diagnosis</h3>
                    <div class="chart-container" id="violinChart"></div>
                </div>
                <div class="viz-card">
                    <h3>MMSE vs ADL Relationship</h3>
                    <div class="chart-container" id="scatterChart"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showSection(id) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
            document.getElementById('section-' + id).classList.add('active');
            document.getElementById('btn-' + id).classList.add('active');
            setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 100);
        }

        const pData = {{ patient | tojson }};
        const avgData = {{ averages | tojson }};
        const statsData = {{ stats | tojson }};

        new Chart(document.getElementById('radarChart'), {
            type: 'radar',
            data: {
                labels: ['MMSE', 'ADL', 'FA', 'BMI', 'Age'],
                datasets: [
                    { label: 'Patient', data: [pData.MMSE/30, pData.ADL/10, pData.FunctionalAssessment/10, pData.BMI/50, pData.Age/100], fill: true, backgroundColor: 'rgba(37, 99, 235, 0.2)', borderColor: '#2563eb' },
                    { label: 'Average', data: [avgData.MMSE/30, avgData.ADL/10, avgData.FunctionalAssessment/10, avgData.BMI/50, avgData.Age/100], fill: true, backgroundColor: 'rgba(148, 163, 184, 0.2)', borderColor: '#94a3b8' }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { r: { min: 0, max: 1, ticks: { display: false } } } }
        });

        const chartData = statsData.mmse_bins.map((b, i) => ({x: b, y: statsData.mmse_counts[i]}));
        new Chart(document.getElementById('mmseDistChart'), {
            type: 'line',
            data: { datasets: [{ label: 'Population', data: chartData, fill: true, backgroundColor: 'rgba(148, 163, 184, 0.2)', borderColor: '#94a3b8', tension: 0.4, pointRadius: 0 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { type: 'linear', min: 0, max: 30 }, y: { display: false } },
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line', xMin: pData.MMSE, xMax: pData.MMSE,
                                borderColor: '#ef4444', borderWidth: 3,
                                label: { content: 'Patient', display: true, position: 'start' }
                            }
                        }
                    }
                }
            }
        });

        const config = { responsive: true, displayModeBar: false };
        Plotly.newPlot('pieChart', {{ fig_pie_json | safe }}, {}, config);
        Plotly.newPlot('sunburstChart', {{ fig_sunburst_json | safe }}, {}, config);
        Plotly.newPlot('heatmapChart', {{ fig_heatmap_json | safe }}, {}, config);
        Plotly.newPlot('violinChart', {{ fig_violin_json | safe }}, {}, config);
        Plotly.newPlot('scatterChart', {{ fig_scatter_json | safe }}, {}, config);
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
        fig_scatter_json=fig_scatter.to_json()
    )

    with open('site_snapshot.html', 'w') as f:
        f.write(html_output)

    print("Responsive interactive snapshot generated: site_snapshot.html")

if __name__ == "__main__":
    generate_full_snapshot()
