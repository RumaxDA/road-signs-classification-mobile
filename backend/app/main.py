from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import detection
from app.database import Base, engine
from app.models.history import DetectionHistory
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"
    FRONTEND_URL: str = "http://localhost:3000"

    class Config: 
        env_file = ".env"
        
settings = Settings()

docs_url = "/docs" if settings.ENVIRONMENT == "dev" else None
redoc_url = "/redoc" if settings.ENVIRONMENT == "dev" else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(title="Road Sign Recognition System", 
              lifespan=lifespan,
              docs_url= docs_url,
              redoc_url=redoc_url)

origins = [
    settings.FRONTEND_URL,
]

if settings.ENVIRONMENT == "dev":
    origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routery
app.include_router(detection.router)

@app.get("/")
def health_check():
    return {"status": "online"}