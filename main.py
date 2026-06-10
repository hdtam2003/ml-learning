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
import data_utils

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
    analysis_data = data_utils.get_analysis_data(DATASET_PATH)
    return templates.TemplateResponse(request=request, name="index.html", context={
        "averages": analysis_data["averages"],
        "total_count": analysis_data["total_count"],
        "charts": analysis_data["charts"]
    })

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

    # Re-fetch analysis data for the updated dashboard
    analysis_data = data_utils.get_analysis_data(DATASET_PATH)

    # We return the same index page but with the updated data
    return templates.TemplateResponse(request=request, name="index.html", context={
        "averages": analysis_data["averages"],
        "total_count": analysis_data["total_count"],
        "charts": analysis_data["charts"],
        "success_msg": "Assessment processed successfully."
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
