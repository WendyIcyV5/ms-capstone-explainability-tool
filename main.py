from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Session, select
from models import Hero, SessionDep, create_db_and_tables
import pickle
import numpy as np
import pandas as pd
import shap
from fastapi import File, UploadFile
import io

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

import time

# CORS (Cross-Origin Resource Sharing) middleware to allow requests from React
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/heroes/")
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero

@app.get("/heroes/")
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes

@app.get("/heroes/{hero_id}")
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero

@app.delete("/heroes/{hero_id}")
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}

@app.post("/explain/")
async def explain_model(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...)
):
    # Load model
    model_bytes = await model_file.read()
    model = pickle.loads(model_bytes)
    
    # Load dataset
    dataset_bytes = await dataset_file.read()
    df = pd.read_csv(io.BytesIO(dataset_bytes))

    # Limit dataset size for performance
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    # Detect model type and use appropriate explainer
    tree_models = (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, DecisionTreeClassifier,
        DecisionTreeRegressor
    )
    linear_models = (
        LogisticRegression, LinearRegression, Ridge, Lasso
    )

    if isinstance(model, tree_models):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, linear_models):
        explainer = shap.LinearExplainer(model, df)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {type(model)}")

    shap_values = explainer.shap_values(df)

    start = time.time()
    shap_values = explainer.shap_values(df)
    elapsed = round(time.time() - start, 2)

    return {
        "feature_names": df.columns.tolist(),
        "shap_values": np.array(shap_values).tolist(),
        "model_type": type(model).__name__,
        "processing_time_seconds": elapsed,
        "samples_processed": len(df)
    }
