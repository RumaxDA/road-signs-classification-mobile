from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/")
async def predict_sign(file: UploadFile = File(...)):
    return JSONResponse(content = {"predicted_sign": "Stop", "confidence": 0.97})



