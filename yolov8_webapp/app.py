from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import cv2
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
model = YOLO('yolov8n.pt')  # You can switch to yolov8s.pt for better accuracy

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    try:
        img_data = base64.b64decode(data['image'].split(',')[1])
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference
    results = model(img)[0]

    # Extract boxes, labels, and confidences
    boxes = results.boxes.xyxy.cpu().tolist()      # [x1, y1, x2, y2]
    classes = results.boxes.cls.cpu().tolist()     # class indices
    scores = results.boxes.conf.cpu().tolist()     # confidence scores

    # Map class indices to names
    names = results.names
    labels = [names[int(c)] for c in classes]

    return jsonify({
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
    print(results[0].boxes)