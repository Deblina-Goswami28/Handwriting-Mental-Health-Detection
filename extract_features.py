import cv2
import numpy as np
import os
import pandas as pd

def extract_features_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (800, 600))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stroke pressure
    stroke_pressure = float(np.mean(thresh))

    # Pen lifts
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_lifts = len(contours)

    # Line spacing
    v_proj = np.sum(thresh, axis=1)
    text_rows = np.where(v_proj > 0)[0]
    line_spacing = (
        float(np.mean(np.diff(text_rows))) if len(text_rows) > 1 else 0.0
    )

    # Slant angle
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 1:
        vx, vy, _, _ = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        slant_angle = float(np.degrees(np.arctan2(vy, vx)))
    else:
        slant_angle = 0.0

    return stroke_pressure, pen_lifts, line_spacing, slant_angle


def create_feature_csv(dataset_dir, output_csv):
    labels = os.listdir(dataset_dir)
    data = []

    for label in labels:
        label_dir = os.path.join(dataset_dir, label)
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_dir, fname)
                features = extract_features_from_image(img_path)
                if features:
                    data.append(list(features) + [label])

    df = pd.DataFrame(data, columns=[
        "stroke_pressure", "pen_lifts", "line_spacing", "slant_angle", "label"
    ])
    df.to_csv(output_csv, index=False)
    print(f"[✔] Features saved to: {output_csv}")


if __name__ == "__main__":
    create_feature_csv("dataset", "handwriting_features.csv")
