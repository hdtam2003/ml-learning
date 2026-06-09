from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from datetime import datetime
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

DATASET_PATH = '1. Dataset .xlsx'

def get_db():
    return pd.read_excel(DATASET_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/submit", response_class=HTMLResponse)
async def submit_assessment(
    request: Request,
    Age: float = Form(...),
    BMI: float = Form(...),
    MMSE: float = Form(...),
    FunctionalAssessment: float = Form(...),
    MemoryComplaints: str = Form(...),
    BehavioralProblems: str = Form(...),
    ADL: float = Form(...)
):
    # Load dataset for comparison
    df = get_db()

    # Calculate population averages for relevant metrics
    averages = {
        "MMSE": round(df['MMSE'].mean(), 2),
        "ADL": round(df['ADL'].mean(), 2),
        "FunctionalAssessment": round(df['FunctionalAssessment'].mean(), 2),
        "BMI": round(df['BMI'].mean(), 2)
    }

    # Create new record
    new_record = {
        "Age": Age,
        "BMI": BMI,
        "MMSE": MMSE,
        "FunctionalAssessment": FunctionalAssessment,
        "MemoryComplaints": int(MemoryComplaints),
        "BehavioralProblems": int(BehavioralProblems),
        "ADL": ADL,
        "Diagnosis": 0, # Default for new entries
        "PatientID": df['PatientID'].max() + 1 if not df.empty else 1,
        "DoctorInCharge": "System Generated"
    }

    # Append to dataset
    new_row = pd.DataFrame([new_record])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_excel(DATASET_PATH, index=False)

    patient_data = {
        "Age": Age,
        "BMI": BMI,
        "MMSE": MMSE,
        "FunctionalAssessment": FunctionalAssessment,
        "MemoryComplaints": MemoryComplaints,
        "BehavioralProblems": BehavioralProblems,
        "ADL": ADL
    }

    return templates.TemplateResponse(request=request, name="results.html", context={
        "patient": patient_data,
        "averages": averages,
        "total_count": len(updated_df),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
