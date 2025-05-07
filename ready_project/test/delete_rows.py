from sqlalchemy.orm import Session
from app.database import SessionLocal, Statistics
import os

def delete_statistics_versions(versions_to_delete):
    db: Session = SessionLocal()
    for version in versions_to_delete:
        stat = db.query(Statistics).filter(Statistics.version == version).first()
        if stat:
            # Удалим файл гистограммы, если существует
            if stat.histogram_path and os.path.exists(stat.histogram_path):
                os.remove(stat.histogram_path)
                print(f"🗑 Удалена гистограмма: {stat.histogram_path}")

            db.delete(stat)
            print(f"🗑 Удалена запись статистики: версия {version}")
    db.commit()
    db.close()

if __name__ == "__main__":
    delete_statistics_versions(["1.0.3", "1.0.4"])
