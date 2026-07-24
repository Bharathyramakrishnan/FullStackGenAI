from fastapi import FastAPI

from app.database.base import Base
from app.database.database import engine

# Import all models so SQLAlchemy knows about them
import app.models

from app.api.v1.user import router as user_router

#"Create every table defined by models that inherit from Base if it doesn't already exist."
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AgentCare AI",
    description="AI Powered Hospital Appointment System",
    version="1.0.0"
)
app.include_router(
    user_router,
    prefix="/api/v1"
)

@app.get("/")
def root():
    return {
        "message": "Welcome to AgentCare AI 🚀"
    }