# app.py
"""
Student Wellness Monitor - FastAPI backend (single-file)
Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime, date
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base
import json
import re

# ----------------- Configuration -----------------
DATABASE_URL = "sqlite:///./checkins.db"  # local SQLite file in the same folder
APP_VERSION = "1.0.0"

# ----------------- Database setup -----------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class CheckinDB(Base):
    __tablename__ = "checkins"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), index=True, nullable=True)
    mood_score = Column(Integer, nullable=False)
    text_entry = Column(Text, nullable=True)
    tags_json = Column(Text, nullable=True)
    sentiment_label = Column(String(32), nullable=True)
    sentiment_score = Column(String(32), nullable=True)
    keywords_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------- Pydantic schemas -----------------
class CheckinIn(BaseModel):
    user_id: Optional[str] = None
    mood_score: int = Field(..., ge=1, le=5)
    text_entry: Optional[str] = ""
    tags: Optional[List[str]] = []

class SentimentOut(BaseModel):
    sentiment: str
    score: float
    keywords: List[str] = []

class CheckinOut(BaseModel):
    id: int
    user_id: Optional[str]
    mood_score: int
    text_entry: Optional[str]
    tags: List[str]
    sentiment: SentimentOut
    created_at: datetime

class StatsPoint(BaseModel):
    day: date
    avg_mood: float
    count: int

# ----------------- Simple sentiment analyzer (rule-based) -----------------
POS_WORDS = {"good","great","happy","joy","relaxed","calm","well","productive","better","excited","relieved","optimistic"}
NEG_WORDS = {"sad","anxious","stressed","angry","worried","tired","depressed","upset","overwhelmed","panic","scared","lonely"}

STOPWORDS = {"i","a","the","and","to","for","of","in","on","is","it","was","am","that","my","me","with","this","today"}

def simple_sentiment_analysis(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {"sentiment": "neutral", "score": 0.0, "keywords": []}
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in clean.split() if t and t not in STOPWORDS]
    if not tokens:
        return {"sentiment": "neutral", "score": 0.0, "keywords": []}
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    score = (pos - neg) / max(1, len(tokens))
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    # keywords: first unique tokens (excluding stopwords)
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
        if len(seen) >= 6:
            break
    return {"sentiment": label, "score": round(score, 3), "keywords": seen}

# ----------------- Recommendations (simple rules) -----------------
def recommend_for_checkin(mood: int, tags: List[str], sentiment: Dict[str, Any]) -> List[Dict[str,str]]:
    recs = []
    if sentiment["sentiment"] == "negative" or mood <= 2:
        recs.append({"title":"Try a 2-minute breathing exercise", "detail":"Box breathing: inhale 4s, hold 4s, exhale 4s — repeat for 2 minutes."})
    if "exam" in tags or "study" in tags:
        recs.append({"title":"Pomodoro study plan", "detail":"Work 25 minutes, break 5 minutes. Repeat 4x, then take a 20-minute break."})
    if "sleep" in tags or ("tired" in sentiment.get("keywords", [])):
        recs.append({"title":"Sleep hygiene tips", "detail":"Keep a consistent sleep schedule and avoid screens 30 minutes before bed."})
    if sentiment["sentiment"] == "positive" and mood >= 4:
        recs.append({"title":"Reinforce good habits", "detail":"Write down one thing that helped today and try to repeat it tomorrow."})
    if not recs:
        recs.append({"title":"Grounding exercise", "detail":"Name 5 things you can see, 4 you can touch, 3 you can hear."})
    return recs[:4]

# ----------------- FastAPI app -----------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Student Wellness Monitor", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status":"ok", "version": APP_VERSION, "time": datetime.utcnow().isoformat()}

@app.post("/api/checkins", response_model=CheckinOut)
def create_checkin(payload: CheckinIn):
    if not (1 <= payload.mood_score <= 5):
        raise HTTPException(status_code=400, detail="mood_score must be 1..5")
    sentiment = simple_sentiment_analysis(payload.text_entry or "")
    now = datetime.utcnow()
    db = SessionLocal()
    try:
        record = CheckinDB(
            user_id = payload.user_id,
            mood_score = payload.mood_score,
            text_entry = payload.text_entry,
            tags_json = json.dumps(payload.tags or []),
            sentiment_label = sentiment["sentiment"],
            sentiment_score = str(sentiment["score"]),
            keywords_json = json.dumps(sentiment.get("keywords", [])),
            created_at = now,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        out = CheckinOut(
            id = record.id,
            user_id = record.user_id,
            mood_score = record.mood_score,
            text_entry = record.text_entry,
            tags = json.loads(record.tags_json or "[]"),
            sentiment = SentimentOut(sentiment=record.sentiment_label or "neutral",
                                     score=float(record.sentiment_score) if record.sentiment_score else 0.0,
                                     keywords=json.loads(record.keywords_json or "[]")),
            created_at = record.created_at
        )
        return out
    finally:
        db.close()

@app.get("/api/checkins", response_model=List[CheckinOut])
def list_checkins(user_id: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=1000), from_dt: Optional[datetime] = None, to_dt: Optional[datetime] = None):
    db = SessionLocal()
    try:
        q = db.query(CheckinDB)
        if user_id:
            q = q.filter(CheckinDB.user_id == user_id)
        if from_dt:
            q = q.filter(CheckinDB.created_at >= from_dt)
        if to_dt:
            q = q.filter(CheckinDB.created_at <= to_dt)
        q = q.order_by(CheckinDB.created_at.asc()).limit(limit)
        results = q.all()
        out = []
        for r in results:
            out.append(CheckinOut(
                id=r.id,
                user_id=r.user_id,
                mood_score=r.mood_score,
                text_entry=r.text_entry,
                tags = json.loads(r.tags_json or "[]"),
                sentiment = SentimentOut(sentiment=r.sentiment_label or "neutral", score=float(r.sentiment_score) if r.sentiment_score else 0.0, keywords=json.loads(r.keywords_json or "[]")),
                created_at=r.created_at
            ))
        return out
    finally:
        db.close()

@app.get("/api/checkins/stats", response_model=List[StatsPoint])
def checkin_stats(user_id: Optional[str] = Query(None), days: int = Query(14, ge=1, le=365)):
    """
    Return a simple daily average mood for the last `days` days (aggregated in Python for DB portability).
    """
    db = SessionLocal()
    try:
        all_rows = db.query(CheckinDB).all()
        from collections import defaultdict
        agg = defaultdict(list)
        for r in all_rows:
            if user_id and r.user_id != user_id:
                continue
            d = r.created_at.date()
            if (datetime.utcnow().date() - d).days < days:
                agg[d].append(r.mood_score)
        points = []
        for d in sorted(agg.keys()):
            vals = agg[d]
            points.append(StatsPoint(day=d, avg_mood=round(sum(vals)/len(vals),3), count=len(vals)))
        return points
    finally:
        db.close()

@app.get("/api/checkins/{checkin_id}/recommendations")
def get_recommendations(checkin_id: int):
    db = SessionLocal()
    try:
        r = db.query(CheckinDB).filter(CheckinDB.id == checkin_id).first()
        if not r:
            raise HTTPException(status_code=404, detail="checkin not found")
        tags = json.loads(r.tags_json or "[]")
        sentiment = {"sentiment": r.sentiment_label or "neutral", "score": float(r.sentiment_score) if r.sentiment_score else 0.0, "keywords": json.loads(r.keywords_json or "[]")}
        recs = recommend_for_checkin(r.mood_score, tags, sentiment)
        return {"recommendations": recs}
    finally:
        db.close()
