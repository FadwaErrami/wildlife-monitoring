import cv2
import numpy as np


def draw_boxes(image: np.ndarray, boxes, scores, color, threshold=0.5, thickness=2, labels=None):
    image_out = image.copy()
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        if score < threshold:
            continue
        x1, y1, x2, y2 = [int(max(0, coord)) for coord in box.tolist()]
        cv2.rectangle(image_out, (x1, y1), (x2, y2), color, thickness)
        
        # Create label with class name if available
        if labels and idx < len(labels):
            label = f"{labels[idx]}: {score:.2f}"
        else:
            label = f"{score:.2f}"
        
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(image_out, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            image_out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image_out
