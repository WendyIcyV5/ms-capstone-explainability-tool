from typing import Optional, Annotated
from datetime import datetime
from sqlmodel import Field, Session, SQLModel, create_engine, JSON, Column
from fastapi import Depends

# PostgreSQL connection URL — format: postgresql://user:password@host:port/database
DATABASE_URL = "postgresql://postgres:010518@localhost:5432/capstone"

# Engine is the connection to the database — created once and reused
engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    """Creates all tables defined as SQLModel classes if they don't already exist"""
    SQLModel.metadata.create_all(engine)

def get_session():
    """
    Yields a database session for each request.
    Using 'with' ensures the session is properly closed after each request,
    preventing connection leaks.
    """
    with Session(engine) as session:
        yield session

# SessionDep is a shortcut type hint used in FastAPI endpoints
# It tells FastAPI to automatically inject a database session into the endpoint
SessionDep = Annotated[Session, Depends(get_session)]


class Upload(SQLModel, table=True):
    """
    Stores metadata about each uploaded model and dataset pair.
    One Upload record is created each time a user submits files.
    """
    # Primary key — auto-incremented by PostgreSQL
    id: Optional[int] = Field(default=None, primary_key=True)
    # Original filename of the uploaded .pkl model file
    model_filename: str
    # Original filename of the uploaded .csv dataset file
    dataset_filename: str
    # Timestamp of when the files were uploaded — defaults to current time
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class ExplainResult(SQLModel, table=True):
    """
    Stores the SHAP explanation results generated for each upload.
    Linked to Upload via upload_id foreign key — one Upload has one ExplainResult.
    """
    # Primary key — auto-incremented by PostgreSQL
    id: Optional[int] = Field(default=None, primary_key=True)
    # Foreign key linking this result back to its corresponding Upload record
    upload_id: int = Field(foreign_key="upload.id")
    # The detected scikit-learn model class name e.g. "RandomForestClassifier"
    model_type: str
    # List of feature names from the dataset — stored as JSON in PostgreSQL
    feature_names: list = Field(sa_column=Column(JSON))
    # Full SHAP values array — stored as JSON in PostgreSQL
    shap_values: list = Field(sa_column=Column(JSON))
    # Number of rows that were actually processed (may be less than total if sampled)
    samples_processed: int
    # Total SHAP computation time in seconds
    processing_time: float
    # Timestamp of when the explanation was generated
    created_at: datetime = Field(default_factory=datetime.utcnow)