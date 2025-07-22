import requests
import base64
import cv2

from face_tracking_api.config import api_url


def crop_and_encode_face(frame, bbox):
    # bbox = [x_center, y_center, width, height]
    x, y, w, h = bbox
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    # Ensure valid crop within image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    face_img = frame[y1:y2, x1:x2]
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', face_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


def get_face_embedding(base64_img):
    payload = {'image': base64_img}
    response = requests.post(f'{api_url}/embedding', json=payload)
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    data = response.json()
    # Defensive: if no face, return None
    embedding = data.get('face', {}).get('embedding')
    if embedding is None:
        raise Exception("No embedding found in response")
    return embedding


def get_person_info(embedding):
    payload = {'embedding': embedding}
    response = requests.post(f'{api_url}/faces/search', json=payload)
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    info = response.json()
    return info['data']['extra_info']  # e.g. {'name': 'Alice', 'email': 'alice@email.com', ...}
