import os
import webbrowser
import subprocess
import dearpygui.dearpygui as dpg
from core.pipeline_runner import run_pipeline
import socket
import subprocess
import os

import socket
import subprocess
import os
import time
import dearpygui.dearpygui as dpg

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def launch_web_server():
    if is_port_in_use(8000):
        dpg.set_value("status", "🌐 Web server is already running.")
        return

    app_path = os.path.join("app", "main.py")
    if not os.path.exists(app_path):
        dpg.set_value("status", "❌ FastAPI entry point not found: app/main.py")
        return

    try:
        process = subprocess.Popen(
            ["uvicorn", "app.main:app", "--reload", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # ждём немного, чтобы дать uvicorn стартануть
        time.sleep(2)

        # проверка снова
        if is_port_in_use(8000):
            dpg.set_value("status", "✅ Web server started.")
        else:
            output = process.stdout.read()
            print("🚨 Ошибка запуска uvicorn:\n", output)
            dpg.set_value("status", "❌ Failed to start web server. Check console.")
    except Exception as e:
        print("🚨 Исключение при запуске uvicorn:", e)
        dpg.set_value("status", f"❌ Exception: {e}")


# --- Callback для выбора DXF и запуска пайплайна
def select_and_run_pipeline_callback(sender, app_data):
    selections = app_data.get("selections", {})
    dxf_files = list(selections.values())

    if not dxf_files:
        dpg.set_value("status", "No DXF files selected.")
        return

    dpg.set_value("status", "Processing files...")
    try:
        run_pipeline(dxf_files)
        dpg.set_value("status", "✅ Comparison complete.")
    except Exception as e:
        print('ошибка тут', e)
        dpg.set_value("status", f"❌ Error: {e}")

# --- Переход в веб-интерфейс
def open_web_interface():
    webbrowser.open("http://127.0.0.1:8000")

# --- GUI Setup
dpg.create_context()
dpg.create_viewport(title="DXF Graph Comparison", width=800, height=400)

# --- Шрифт (если есть)
with dpg.font_registry():
    font_path = os.path.join("static", "fonts", "Roboto-Regular.ttf")
    if os.path.exists(font_path):
        default_font = dpg.add_font(font_path, 18)
        dpg.bind_font(default_font)

# --- Главное окно
with dpg.window(tag="main_window", width=800, height=400):
    dpg.add_text("Select DXF files for comparison:")
    dpg.add_button(label="Choose DXF Files", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_text("", tag="status")
    dpg.add_spacer(height=10)
    dpg.add_button(label="Open Web Interface", callback=open_web_interface)

# --- Диалог выбора файлов
with dpg.file_dialog(tag="file_dialog_id", width=700, height=400,
                     show=False, callback=select_and_run_pipeline_callback,
                     file_count=0, directory_selector=False):
    dpg.add_file_extension(".dxf", color=(0, 200, 100, 255))

# --- Запуск
launch_web_server()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
