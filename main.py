from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile
from sqlmodel import Session, select
from models import Upload, ExplainResult, SessionDep, create_db_and_tables, get_session
import pickle
import numpy as np
import pandas as pd
import shap
import io
import time

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

# CORS middleware allows React (port 5173) to communicate with FastAPI (port 8000)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Create database tables on startup if they don't exist"""
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/explain/")
async def explain_model(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    # Load the uploaded .pkl model file into a Python object
    model_bytes = await model_file.read()
    model = pickle.loads(model_bytes)

    # Load the uploaded .csv dataset into a pandas DataFrame
    dataset_bytes = await dataset_file.read()
    df = pd.read_csv(io.BytesIO(dataset_bytes))

    # Limit dataset size for performance — SHAP is computationally expensive
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    # Supported tree-based model types — use TreeExplainer
    tree_models = (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, DecisionTreeClassifier,
        DecisionTreeRegressor
    )
    # Supported linear model types — use LinearExplainer
    linear_models = (
        LogisticRegression, LinearRegression, Ridge, Lasso
    )

    # Detect model type and route to correct SHAP explainer
    if isinstance(model, tree_models):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, linear_models):
        explainer = shap.LinearExplainer(model, df)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {type(model)}"
        )

    # Run SHAP and measure processing time
    start = time.time()
    shap_values = explainer.shap_values(df)
    elapsed = round(time.time() - start, 2)

    feature_names = df.columns.tolist()
    shap_list = np.array(shap_values).tolist()
    model_type = type(model).__name__
    samples_processed = len(df)

    # Save upload metadata to database
    upload = Upload(
        model_filename=model_file.filename,
        dataset_filename=dataset_file.filename
    )
    session.add(upload)
    session.commit()
    session.refresh(upload)

    # Save SHAP results to database linked to the upload
    result = ExplainResult(
        upload_id=upload.id,
        model_type=model_type,
        feature_names=feature_names,
        shap_values=shap_list,
        samples_processed=samples_processed,
        processing_time=elapsed
    )
    session.add(result)
    session.commit()

    return {
        "feature_names": feature_names,
        "shap_values": shap_list,
        "model_type": model_type,
        "processing_time_seconds": elapsed,
        "samples_processed": samples_processed
    }