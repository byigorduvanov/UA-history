from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class SpreadPoint(BaseModel):
    """Модель для точки спреду з API"""
    model_config = ConfigDict(populate_by_name=True)
    
    t: int  # timestamp
    in_: float = Field(alias="in")  # значення in
    out: float  # значення out
    lf: float  # left fee
    rf: float  # right fee


class FundingPoint(BaseModel):
    """Модель для точки funding"""
    t: int  # timestamp
    r: float  # rate


class FundingData(BaseModel):
    """Модель для funding даних"""
    left: List[FundingPoint]
    right: List[FundingPoint]


class RangeData(BaseModel):
    """Модель для діапазону даних"""
    model_config = ConfigDict(populate_by_name=True)
    
    from_: int = Field(alias="from")  # початковий timestamp
    to: int  # кінцевий timestamp


class HistoricalData(BaseModel):
    """Модель для відповіді API з історичними даними"""
    spreads: List[SpreadPoint]
    funding: FundingData
    range: RangeData


class APIResponse(BaseModel):
    """Модель для повної відповіді API"""
    data: HistoricalData

