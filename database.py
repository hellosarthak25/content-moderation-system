"""
database.py
SQLAlchemy Database Layer for Content Moderation Audit Logging.
Uses SQLite for zero-configuration local persistence.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, desc
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = "sqlite:///moderation_audit.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    comment_text = Column(String, nullable=False)
    verdict = Column(String, nullable=False)  # APPROVED, FLAGGED_FOR_REVIEW, AUTO_BLOCKED
    primary_category = Column(String, default="clean")
    
    # Probability scores for the 6 labels
    score_toxic = Column(Float, default=0.0)
    score_severe_toxic = Column(Float, default=0.0)
    score_obscene = Column(Float, default=0.0)
    score_threat = Column(Float, default=0.0)
    score_insult = Column(Float, default=0.0)
    score_identity_hate = Column(Float, default=0.0)
    
    latency_ms = Column(Float, default=0.0)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "comment_text": self.comment_text,
            "verdict": self.verdict,
            "primary_category": self.primary_category,
            "scores": {
                "toxic": round(self.score_toxic, 4),
                "severe_toxic": round(self.score_severe_toxic, 4),
                "obscene": round(self.score_obscene, 4),
                "threat": round(self.score_threat, 4),
                "insult": round(self.score_insult, 4),
                "identity_hate": round(self.score_identity_hate, 4),
            },
            "latency_ms": round(self.latency_ms, 2),
        }

# Initialize database schema
Base.metadata.create_all(bind=engine)

def log_comment(comment_text: str, verdict: str, primary_category: str, scores: dict, latency_ms: float):
    """Saves a moderation record into the database."""
    session = SessionLocal()
    try:
        record = ModerationLog(
            comment_text=comment_text,
            verdict=verdict,
            primary_category=primary_category,
            score_toxic=scores.get("toxic", 0.0),
            score_severe_toxic=scores.get("severe_toxic", 0.0),
            score_obscene=scores.get("obscene", 0.0),
            score_threat=scores.get("threat", 0.0),
            score_insult=scores.get("insult", 0.0),
            score_identity_hate=scores.get("identity_hate", 0.0),
            latency_ms=latency_ms,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.to_dict()
    finally:
        session.close()

def get_recent_logs(limit: int = 50):
    """Fetches recent moderation audit logs."""
    session = SessionLocal()
    try:
        logs = session.query(ModerationLog).order_by(desc(ModerationLog.timestamp)).limit(limit).all()
        return [log.to_dict() for log in logs]
    finally:
        session.close()

def get_statistics():
    """Computes overall moderation metrics for the dashboard."""
    session = SessionLocal()
    try:
        total = session.query(ModerationLog).count()
        if total == 0:
            return {
                "total": 0,
                "approved": 0,
                "flagged": 0,
                "blocked": 0,
                "avg_latency_ms": 0.0,
            }
        
        approved = session.query(ModerationLog).filter(ModerationLog.verdict == "APPROVED").count()
        flagged = session.query(ModerationLog).filter(ModerationLog.verdict == "FLAGGED_FOR_REVIEW").count()
        blocked = session.query(ModerationLog).filter(ModerationLog.verdict == "AUTO_BLOCKED").count()
        
        # Calculate average latency
        avg_latency = session.query(ModerationLog.latency_ms).all()
        avg_lat = sum(l[0] for l in avg_latency) / total if avg_latency else 0.0
        
        return {
            "total": total,
            "approved": approved,
            "flagged": flagged,
            "blocked": blocked,
            "approved_pct": round((approved / total) * 100, 1),
            "flagged_pct": round((flagged / total) * 100, 1),
            "blocked_pct": round((blocked / total) * 100, 1),
            "avg_latency_ms": round(avg_lat, 2),
        }
    finally:
        session.close()

def clear_all_logs():
    """Wipes all moderation logs from the database."""
    session = SessionLocal()
    try:
        session.query(ModerationLog).delete()
        session.commit()
        return {"status": "success", "message": "All moderation logs cleared."}
    finally:
        session.close()
