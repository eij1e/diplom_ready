from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI
from app.routers import router  # Убедись, что файл routers.py находится в app/

app = FastAPI()

# Подключаем маршруты
app.include_router(router)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
