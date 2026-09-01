from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    flight_id: str
    phases: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    frames_dir: str | None = None
    poses_dir: str | None = None
    mock: bool = False
