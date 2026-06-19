from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VideoMetadata(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_hash = Column(String, unique=True, index=True, nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()