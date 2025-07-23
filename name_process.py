import requests
import base64
import cv2
from face_tracking_api.config import api_url
from config import auth_type, embeddings, find_face, api_url, headers
import ast


def crop_and_encode_face(frame, bbox):
    # bbox = [x_center, y_center, width, height]
    # print(bbox)
    x, y, w, h = bbox
    x1, y1 = int(x - w * 2/3), int(y - h)
    x2, y2 = int(x + w * 2/3), int(y + h)
    # Ensure valid crop within image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    face_img = frame[y1:y2, x1:x2]
    #DEBUG -- done, they are working fine
    '''
    cv2.imwrite(f"debug_crop_{x1}_{y1}_{x2}_{y2}.jpg", face_img)
    '''
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', face_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


def get_face_embedding(base64_img):
    payload = {'image': base64_img, "type": 0}
    response = requests.post(f'{api_url}/{embeddings}', json=payload, headers=headers, verify=False)
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    response = response.json()
    # print(response)
    # Defensive: if no face, return None
    # embedding = data.get('face', {}).get('embedding')
    if 'face' not in response:
        # print(f"Face not detected: {response}")
        return None  # Or handle however you need
    embedding = response['face']['embedding']
    # print(embedding)
    return embedding

def get_person_info(embedding):
    if embedding is None:
        return None
    payload = {"group": "test",
               "threshold": 63.0,
               "limit":1,
               'embedding': embedding}
    response = requests.post(f'{api_url}/{find_face}', json=payload, headers=headers, verify=False)
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    info = response.json()
    data = info.get('data', [])
    if not data or not isinstance(data, list) :
        return None
    candidate = data[0]
    extra_info_str = candidate.get('extra_info', None)
    if not extra_info_str:
        return None
    try:
        extra_info = ast.literal_eval(extra_info_str)
        # Optionally, add probability info as well:
        extra_info['prob'] = candidate.get('prob', None)
        return extra_info
    except Exception as e:
        print(f"Error parsing extra_info: {e}")
        return None
