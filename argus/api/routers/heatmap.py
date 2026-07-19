from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from argus.api.auth import get_current_user
from argus.api.inference_pipeline import _GRID_COLS, _GRID_ROWS, registry
from argus.api.schemas import HeatmapResponse

router = APIRouter(prefix="/api/v1/heatmap", tags=["heatmap"])


@router.get("/{camera_id}", response_model=HeatmapResponse)
async def get_heatmap(camera_id: int, _: dict = Depends(get_current_user)):
    grid = registry.get_heatmap(camera_id)
    if grid is None:
        # Return empty grid if pipeline is not running
        grid = [[0.0] * _GRID_COLS for _ in range(_GRID_ROWS)]
    return HeatmapResponse(
        camera_id=camera_id,
        grid=grid,
        grid_rows=_GRID_ROWS,
        grid_cols=_GRID_COLS,
        timestamp=datetime.utcnow(),
    )
