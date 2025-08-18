from fastapi import FastAPI
from app.signs import schemas as signs_schemas
from app.predictions import schemas as predicions_schemas
from app.signs import routes as signs_routes
from app.predictions import routes as predictions_routes
from fastapi.staticfiles import StaticFiles

app = FastAPI(title = "Road Sign Classification API",
              version="1.0.0")

app.mount("/static", StaticFiles(directory = "app/static"), name = "static")

app.include_router(signs_routes.router, prefix="/signs", tags=["Signs"])
app.include_router(predictions_routes.router, prefix="/predictions", tags=["Predictions"])

