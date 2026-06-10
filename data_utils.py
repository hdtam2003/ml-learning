import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

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

def to_json_clean(fig):
    return json.dumps(fig.to_dict(), cls=NpEncoder)

def get_analysis_data(dataset_path):
    df = pd.read_excel(dataset_path)

    # Preprocessing
    numeric_cols_to_fix = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment', 'ADL', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'Diagnosis']
    for col in numeric_cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=['Diagnosis', 'MMSE', 'ADL']).copy()
    df_clean['Diagnosis_Label'] = df_clean['Diagnosis'].map({0: 'Healthy', 1: 'Alzheimer\'s'}).fillna('Unknown')
    df_clean['Gender_Label'] = df_clean['Gender'].map({0: 'Male', 1: 'Female'}).fillna('Unknown')

    # Education Level mapping for EDA
    edu_map = {0: 'None', 1: 'High School', 2: 'Bachelor', 3: 'Higher'}
    df_clean['Education_Label'] = df_clean['EducationLevel'].map(edu_map).fillna('Unknown')

    # Averages
    averages = {
        "MMSE": round(df_clean['MMSE'].mean(), 2),
        "ADL": round(df_clean['ADL'].mean(), 2),
        "FunctionalAssessment": round(df_clean['FunctionalAssessment'].mean(), 2),
        "BMI": round(df_clean['BMI'].mean(), 2),
        "Age": round(df_clean['Age'].mean(), 2)
    }

    # Plotly Charts
    # 1. Diagnosis Pie
    fig_pie = px.pie(df_clean, names='Diagnosis_Label', color='Diagnosis_Label',
                    color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_pie.update_layout(title="Population Diagnosis Prevalence", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 2. Demographic Sunburst
    fig_sunburst = px.sunburst(df_clean, path=['Gender_Label', 'Diagnosis_Label'],
                              color='Diagnosis_Label',
                              color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_sunburst.update_layout(title="Demographic Distribution", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 3. Age Violin
    fig_violin = px.violin(df_clean, y="Age", x="Diagnosis_Label", color="Diagnosis_Label",
                          box=True, points=None,
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_violin.update_layout(title="Age Distribution", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 3b. BMI Violin
    fig_bmi = px.violin(df_clean, y="BMI", x="Diagnosis_Label", color="Diagnosis_Label",
                          box=True, points=None,
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_bmi.update_layout(title="BMI Distribution", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 4. MMSE vs ADL Scatter
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
                             xaxis_title="MMSE Score",
                             yaxis_title="ADL Score",
                             font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 5. Lifestyle Factors Bar (Normalized for comparison)
    lifestyle_cols = ['Smoking', 'AlcoholConsumption', 'PhysicalActivity']
    lifestyle_df = df_clean.groupby('Diagnosis_Label')[lifestyle_cols].mean().reset_index()

    # Normalize: Smoking is already 0-1 (prevalence), Alcohol 0-20 -> /20, PhysicalActivity 0-10 -> /10
    lifestyle_df['AlcoholConsumption'] = lifestyle_df['AlcoholConsumption'] / 20
    lifestyle_df['PhysicalActivity'] = lifestyle_df['PhysicalActivity'] / 10

    lifestyle_melted = lifestyle_df.melt(id_vars='Diagnosis_Label', var_name='Factor', value_name='Score')
    fig_lifestyle = px.bar(lifestyle_melted, x='Factor', y='Score', color='Diagnosis_Label', barmode='group',
                          labels={'Score': 'Impact Score (Normalized 0-1)'},
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_lifestyle.update_layout(title="Lifestyle Factors Comparison (Normalized)", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 6. Education Histogram
    fig_edu = px.histogram(df_clean, x="Education_Label", color="Diagnosis_Label", barmode='group',
                          category_orders={"Education_Label": ["None", "High School", "Bachelor", "Higher"]},
                          color_discrete_map={'Healthy': '#10b981', 'Alzheimer\'s': '#ef4444', 'Unknown': '#94a3b8'})
    fig_edu.update_layout(title="Education Level vs Diagnosis", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    # 7. Correlation Heatmap
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
    fig_heatmap.update_layout(title="Clinical Metric Correlation", font_family="Inter", margin=dict(t=50, b=20, l=20, r=20))

    return {
        "averages": averages,
        "total_count": len(df_clean),
        "charts": {
            "pie": to_json_clean(fig_pie),
            "sunburst": to_json_clean(fig_sunburst),
            "violin": to_json_clean(fig_violin),
            "bmi": to_json_clean(fig_bmi),
            "scatter": to_json_clean(fig_scatter),
            "lifestyle": to_json_clean(fig_lifestyle),
            "edu": to_json_clean(fig_edu),
            "heatmap": to_json_clean(fig_heatmap)
        }
    }
