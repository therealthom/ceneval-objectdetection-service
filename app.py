from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import ast

app = Flask(__name__)
CORS(app)

# Global variables for settings
current_model = 'yolov10m.pt'
detection_interval = 5

# Dictionary to store loaded models
loaded_models = {}


def load_model(model_name):
    if model_name not in loaded_models:
        loaded_models[model_name] = YOLO(model_name)
    return loaded_models[model_name]


def init_db():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  image TEXT,
                  objects TEXT)''')
    conn.commit()
    conn.close()


init_db()


@app.template_filter('format_time')
def format_time(timestamp):
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@app.template_filter('parse_objects')
def parse_objects(objects_str):
    objects = ast.literal_eval(objects_str)
    return [{'class': obj['class'], 'confidence': obj['confidence']} for obj in objects]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    image_data = request.json['image']
    model_name = current_model  # Use the global current_model

    # Remove the base64 prefix from the string
    image_data = image_data.split(',')[1]

    # Decode the image
    image = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

    # Load the specified model
    model = load_model(model_name)

    # Perform detection
    results = model(image)

    # Process the results
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

    # Draw bounding boxes on the image
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{obj['class']} {obj['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert the processed image back to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Check if there's an anomaly
    if is_anomaly(detected_objects):
        save_anomaly(image_base64, detected_objects)

    return jsonify({
        "objects": detected_objects,
        "image": image_base64
    })


def is_anomaly(objects):
    person_count = sum(1 for obj in objects if obj['class'].lower() == 'person')
    return person_count != 1 or any(obj['class'].lower() != 'person' for obj in objects)


def save_anomaly(image, objects):
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("INSERT INTO anomalies (timestamp, image, objects) VALUES (?, ?, ?)",
              (datetime.now().isoformat(), image, str(objects)))
    conn.commit()
    conn.close()


@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("SELECT * FROM anomalies ORDER BY timestamp DESC LIMIT 10")
    anomalies = c.fetchall()
    conn.close()

    yolo_models = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt']
    return render_template('dashboard.html', anomalies=anomalies,
                           current_model=current_model,
                           detection_interval=detection_interval,
                           yolo_models=yolo_models)


@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model
    model = request.json['model']
    current_model = model
    return jsonify({"message": f"Model set to {model}"})


@app.route('/set_interval', methods=['POST'])
def set_interval():
    global detection_interval
    interval = request.json['interval']
    detection_interval = int(interval)
    return jsonify({"message": f"Interval set to {interval} seconds"})


@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify({
        "current_model": current_model,
        "detection_interval": detection_interval
    })


@app.route('/clear_database', methods=['POST'])
def clear_database():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute("DELETE FROM anomalies")
    conn.commit()
    conn.close()
    return jsonify({"message": "Database cleared successfully"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)