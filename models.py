from __future__ import annotations

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class Price(BaseModel):
    amount: Optional[int] = None
    currency: Optional[str] = None


class Listing(BaseModel):
    url: str
    source: str
    listing_id: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    address_line: Optional[str] = None
    price: Price = Field(default_factory=Price)
    area_m2: Optional[int] = None
    land_area_m2: Optional[int] = None
    rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[int] = None
    floors_total: Optional[int] = None
    description: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    photos: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    summary_ru: Optional[str] = None
    raw_meta: Dict[str, Any] = Field(default_factory=dict)
