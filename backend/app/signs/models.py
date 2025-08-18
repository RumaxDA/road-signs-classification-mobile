from pydantic import BaseModel

class TrafficSign(BaseModel):
    id: int
    name: str
    description: str | None = None
    image_url: str | None = None