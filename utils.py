import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load full-body classifier
script_dir = os.path.dirname(os.path.abspath(__file__))
body_cascade_path = os.path.join(script_dir, 'haarcascade_fullbody.xml')
body_cascade = cv2.CascadeClassifier(body_cascade_path)
if body_cascade.empty():
    raise ValueError("Could not load body cascade. Ensure haarcascade_fullbody.xml is in same directory.")

def extract_person(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Step 1: Try detecting full body
    bodies = body_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)
    if len(bodies) > 0:
        x, y, w, h = max(bodies, key=lambda rect: rect[2] * rect[3])
        rect = (
            max(0, x - int(w * 0.2)),
            max(0, y - int(h * 0.2)),
            min(w + int(w * 0.4), img.shape[1] - x),
            min(h + int(h * 0.4), img.shape[0] - y)
        )
    else:
        # Step 2: Fallback to face detection
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            rect = (
                max(0, x - int(w * 0.5)),
                max(0, y - int(h * 0.5)),
                min(w * 2, img.shape[1] - x),
                min(h * 6, img.shape[0] - y)
            )
        else:
            # Fallback: center crop
            h_img, w_img = img.shape[:2]
            rect = (w_img // 6, h_img // 6, w_img * 2 // 3, h_img * 2 // 3)

    # Step 3: Apply GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Step 4: Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(mask2, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=3)

    # Step 5: Remove noise
    mask_cleaned = cv2.medianBlur(mask2 * 255, 5)
    mask_cleaned = cv2.threshold(mask_cleaned, 127, 255, cv2.THRESH_BINARY)[1]

    # Step 6: Convert to transparent BGRA
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_bgra[:, :, 3] = mask_cleaned

    return img_bgra


def blend_images(person, background):
    h, w = background.shape[:2]
    person_rgb = person[:, :, :3]
    mask = person[:, :, 3]

    if mask.max() == 0:
        raise ValueError("Empty mask. Person not detected properly.")

    # Resize person image
    aspect = person_rgb.shape[1] / person_rgb.shape[0]
    new_h = h // 4
    new_w = int(new_h * aspect)
    if new_w > w:
        new_w = w
        new_h = int(new_w / aspect)

    person_rgb = cv2.resize(person_rgb, (new_w, new_h))
    mask_resized = cv2.resize(mask, (new_w, new_h))

    # Coordinates to place person
    bottom_margin = h // 6
    y_offset = h - bottom_margin - new_h
    x_offset = (w - new_w) // 2
    if y_offset < 0:
        y_offset = 0

    person_canvas = np.zeros_like(background)
    mask_canvas = np.zeros((h, w), dtype=np.uint8)

    person_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = person_rgb
    mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized

    # Alpha blending
    alpha = mask_canvas.astype(float) / 255.0
    alpha = cv2.merge([alpha] * 3)

    foreground = person_canvas.astype(float)
    background = background.astype(float)

    blended = cv2.multiply(alpha, foreground) + cv2.multiply(1.0 - alpha, background)
    return blended.astype(np.uint8)
