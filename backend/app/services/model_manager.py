import logging
from app.core.config import settings
from app.services.traffic_logic import TrafficSignSystem
from enum import Enum

logger = logging.getLogger(__name__)

class ModelVersion(str, Enum):
    CNN_48 = "CNN_48_v1"
    TL_224 = "TL_224_v1"
    TL_96 = "TL_96_v1"

class ModelManager:
    def __init__(self):
        self._models = {}
        self._initialize_pool()

    def _initialize_pool(self):
        logger.info("Inicjalizacja puli modeli ML...")
        try:
            self._models = {
                "CNN_48_v1": TrafficSignSystem(
                    settings.YOLO_MODEL_PATH, 
                    settings.CNN_MODEL_PATH, 
                    img_size=48
                ),
                "TL_224_v1" : TrafficSignSystem(
                    settings.YOLO_MODEL_PATH,
                    settings.TL_MODEL_PATH,
                    img_size=224
                ),
                "TL_96_v1" : TrafficSignSystem(
                    settings.YOLO_MODEL_PATH,
                    settings.TL_96_MODEL_PATH,
                    img_size=96
                )
            }
            logger.info(f"Załadowano modele: {list(self._models.keys())}")
        except Exception as e:
            logger.error(f"Błąd podczas ładowania modeli: {e}")
            raise e

    def get_model(self, model_version: ModelVersion) -> TrafficSignSystem:
        return self._models.get(model_version)

model_manager = ModelManager()