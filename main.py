from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import os
import threading

app = FastAPI()

# Thread lock for Excel file access
excel_lock = threading.Lock()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

DATASET_PATH = '1. Dataset .xlsx'

class AssessmentData(BaseModel):
    Age: float = Field(..., ge=0, le=120)
    BMI: float = Field(..., ge=10, le=50)
    MMSE: float = Field(..., ge=0, le=30)
    FunctionalAssessment: float = Field(..., ge=0, le=10)
    MemoryComplaints: int = Field(..., ge=0, le=1)
    BehavioralProblems: int = Field(..., ge=0, le=1)
    ADL: float = Field(..., ge=0, le=10)

def get_db():
    with excel_lock:
        return pd.read_excel(DATASET_PATH)

def save_db(df):
    with excel_lock:
        df.to_excel(DATASET_PATH, index=False)

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
    MemoryComplaints: int = Form(...),
    BehavioralProblems: int = Form(...),
    ADL: float = Form(...)
):
    # Pydantic validation
    try:
        data = AssessmentData(
            Age=Age, BMI=BMI, MMSE=MMSE,
            FunctionalAssessment=FunctionalAssessment,
            MemoryComplaints=MemoryComplaints,
            BehavioralProblems=BehavioralProblems,
            ADL=ADL
        )
    except Exception as e:
        return HTMLResponse(content=f"Validation Error: {e}", status_code=400)

    # Load dataset for comparison
    df = get_db()

    # Calculate population averages for relevant metrics
    averages = {
        "MMSE": round(df['MMSE'].mean(), 2),
        "ADL": round(df['ADL'].mean(), 2),
        "FunctionalAssessment": round(df['FunctionalAssessment'].mean(), 2),
        "BMI": round(df['BMI'].mean(), 2),
        "Age": round(df['Age'].mean(), 2)
    }

    # Create new record
    new_record = {
        "Age": data.Age,
        "BMI": data.BMI,
        "MMSE": data.MMSE,
        "FunctionalAssessment": data.FunctionalAssessment,
        "MemoryComplaints": data.MemoryComplaints,
        "BehavioralProblems": data.BehavioralProblems,
        "ADL": data.ADL,
        "Diagnosis": 0, # Default for new entries
        "PatientID": int(df['PatientID'].max() + 1) if not df.empty else 1,
        "DoctorInCharge": "System Generated"
    }

    # Append to dataset
    new_row = pd.DataFrame([new_record])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    save_db(updated_df)

    # Statistics for distribution charts
    mmse_hist, mmse_edges = np.histogram(updated_df['MMSE'].dropna(), bins=20)
    adl_hist, adl_edges = np.histogram(updated_df['ADL'].dropna(), bins=20)

    stats = {
        "mmse_bins": [round(b, 1) for b in mmse_edges[:-1]],
        "mmse_counts": mmse_hist.tolist(),
        "adl_bins": [round(b, 1) for b in adl_edges[:-1]],
        "adl_counts": adl_hist.tolist()
    }

    patient_data = data.model_dump()
    # Convert MemoryComplaints/BehavioralProblems back to Yes/No for display if needed
    # but the template currently uses the raw values.
    # Results template expects string "Yes"/"No" for the cards in some versions but the previous logic sent strings.
    # Let's check results.html cards.

    patient_display = patient_data.copy()
    patient_display["MemoryComplaints"] = "Yes" if data.MemoryComplaints == 1 else "No"
    patient_display["BehavioralProblems"] = "Yes" if data.BehavioralProblems == 1 else "No"

    return templates.TemplateResponse(request=request, name="results.html", context={
        "patient": patient_display,
        "averages": averages,
        "stats": stats,
        "total_count": len(updated_df),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
