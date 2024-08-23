# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de requisitos e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Descargar los modelos YOLOv8 (ajusta según los modelos que necesites)
RUN python -c "from ultralytics import YOLO; YOLO('yolov10n.pt'); YOLO('yolov10s.pt'); YOLO('yolov10m.pt'); YOLO('yolov10l.pt'); YOLO('yolov10x.pt')"

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]