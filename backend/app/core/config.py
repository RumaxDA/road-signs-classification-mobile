import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))

class Settings(BaseSettings):
    # Dane bazy
    POSTGRES_USER: str = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD: str = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_DB: str = os.environ["POSTGRES_DB"]
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: int = os.getenv("POSTGRES_PORT", 5432)

    YOLO_MODEL_PATH: str = "models_ai/best.pt"
    CNN_MODEL_PATH: str = "models_ai/cnn_48_v1.keras"
    TL_MODEL_PATH: str = "models_ai/efficientnet_b0_224_v1.keras"
    TL_96_MODEL_PATH: str = "models_ai/efficientnet_b0_96_v1.keras"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Konfiguracja ładowania
    model_config = SettingsConfigDict(
        env_file= ".env",
        extra="ignore",
        env_file_encoding="utf-8",
        case_sensitive= True
    )

settings = Settings()

if __name__ == "__main__":
    pass