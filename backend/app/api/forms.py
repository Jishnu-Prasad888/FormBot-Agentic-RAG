from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.graph.form_recommender import recommend_forms


class FormRecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)


router = APIRouter(prefix="/api/forms", tags=["forms"])


@router.post("/recommend")
async def recommend(req: FormRecommendRequest) -> dict[str, Any]:
    return await recommend_forms(req.query)
