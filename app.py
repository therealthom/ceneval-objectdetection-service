from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)
CORS(app)

# Diccionario para almacenar los modelos cargados
loaded_models = {}


def load_model(model_name):
    if model_name not in loaded_models:
        loaded_models[model_name] = YOLO(model_name)
    return loaded_models[model_name]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    image_data = request.json['image']
    model_name = request.json['model']

    # Eliminar el prefijo de la cadena base64
    image_data = image_data.split(',')[1]

    # Decodificar la imagen
    image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

    # Cargar el modelo especificado
    model = load_model(model_name)

    # Realizar la detección
    results = model(image)

    # Procesar los resultados
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls)
            conf = float(box.conf)
            detected_objects.append({
                "class": r.names[class_id],
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    # Dibujar los bounding boxes en la imagen
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{obj['class']} {obj['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convertir la imagen procesada de vuelta a base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "objects": detected_objects,
        "image": image_base64
    })


@app.route('/set_interval', methods=['POST'])
def set_interval():
    interval = request.json['interval']
    # Aquí puedes implementar la lógica para manejar el intervalo si es necesario
    return jsonify({"message": f"Interval set to {interval} seconds"})


if __name__ == '__main__':
    app.run(debug=True)