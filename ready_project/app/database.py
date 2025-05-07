import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, '..', 'data', 'comparisons.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ComparisonResult(Base):
    __tablename__ = "comparison_results"
    id = Column(Integer, primary_key=True, index=True)
    graph_a = Column(String, index=True)
    graph_b = Column(String, index=True)
    ransac_predict = Column(Float)
    gcn_predict = Column(Integer)
    gin_predict = Column(Integer)
    image_path = Column(String)
    version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_plagiarism = Column(Boolean, default=False)

class VersionInfo(Base):
    __tablename__ = "version_info"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=0)  # 1 = активная версия


class Statistics(Base):
    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, index=True)
    ransac_count_over_0_5 = Column(Integer)
    gcn_positive = Column(Integer)
    gin_positive = Column(Integer)
    ransac_percentile_75 = Column(Float)
    ransac_percentile_90 = Column(Float)
    histogram_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
