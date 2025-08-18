from fastapi import APIRouter
from .models import TrafficSign

router = APIRouter()

signs = [
    TrafficSign(id = 1,name = "Speed limit (20km/h)", image_url = "/static/0.png"),
    TrafficSign(id = 2,name = "Speed limit (40km/h)", image_url = "/static/1.png"),
    TrafficSign(id = 3,name = "Speed limit (60km/h)", image_url = "/static/2.png"),
]

@router.get("/")
def get_all_signs():
    return signs

@router.get("/{sign_id}", response_model=TrafficSign)
def get_sign(sign_id: int):
    return next(sign for sign in signs if sign.id == sign_id)
