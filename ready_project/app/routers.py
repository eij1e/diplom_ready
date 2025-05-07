import os
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.database import SessionLocal, ComparisonResult, VersionInfo, Statistics
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Настройка шаблонов
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

router = APIRouter()

# Dependency — для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Главная страница с фильтрацией по версии ---
@router.get("/")
def index(request: Request, version: str = "", db: Session = Depends(get_db)):
    query = db.query(ComparisonResult)

    if version:
        query = query.filter(ComparisonResult.version == version)

    comparisons = query.order_by(ComparisonResult.created_at.desc()).all()
    versions = db.query(VersionInfo).order_by(VersionInfo.created_at.desc()).all()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "comparisons": comparisons,
        "versions": versions,
        "selected_version": version
    })


# === Страница управления версиями ===
@router.get("/versions")
def manage_versions(request: Request, db: Session = Depends(get_db)):
    versions = db.query(VersionInfo).order_by(VersionInfo.created_at.desc()).all()
    return templates.TemplateResponse("versions.html", {"request": request, "versions": versions})

# === Удаление версии (и записей, и изображений) ===
@router.post("/delete_version")
def delete_version(version: str = Form(...), db: Session = Depends(get_db)):
    # Удаляем изображения
    comparisons = db.query(ComparisonResult).filter(ComparisonResult.version == version).all()
    for comp in comparisons:
        if comp.image_path and os.path.exists(comp.image_path):
            os.remove(comp.image_path)

    # Удаляем гистограмму статистики
    stat = db.query(Statistics).filter(Statistics.version == version).first()
    if stat and stat.histogram_path and os.path.exists(stat.histogram_path):
        os.remove(stat.histogram_path)

    # Удаляем записи из всех таблиц
    db.query(ComparisonResult).filter(ComparisonResult.version == version).delete()
    db.query(VersionInfo).filter(VersionInfo.version == version).delete()
    db.query(Statistics).filter(Statistics.version == version).delete()
    db.commit()

    return RedirectResponse(url="/versions", status_code=303)


@router.get("/stats")
def stats_page(request: Request, db: Session = Depends(get_db)):
    stats = db.query(Statistics).order_by(Statistics.version.desc()).all()
    return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})
