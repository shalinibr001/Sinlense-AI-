# preprocess.py
import cv2
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

# ---------------------------------------------
# Read image as RGB and resize
# ---------------------------------------------
def read_rgb(path_or_array, target=(224,224)):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = path_or_array.copy()
    img = cv2.resize(img, target)
    return img


# ---------------------------------------------
# Prepare image for CNN model
# ---------------------------------------------
def preprocess_for_model(img_rgb, target_size=(224,224)):
    img = cv2.resize(img_rgb, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ---------------------------------------------
# Lesion segmentation using Otsu threshold
# ---------------------------------------------
def segment_lesion(img_rgb, out_size=(224,224)):
    img = cv2.resize(img_rgb, (300,300))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    try:
        thresh = threshold_otsu(blur)
        mask = (blur < thresh).astype("uint8") * 255
    except:
        _, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)

    if not props:
        return cv2.resize(mask, out_size)

    largest = max(props, key=lambda p: p.area)
    lesion_mask = (labels == largest.label).astype("uint8") * 255
    lesion_mask = cv2.resize(lesion_mask, out_size)
    return lesion_mask


# ---------------------------------------------
# Compute ABCD Features
# ---------------------------------------------
def compute_abcd(img_rgb):
    img = cv2.resize(img_rgb, (224,224))
    mask = segment_lesion(img, out_size=(224,224))

    _, m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"A": 0.0, "B": 0.0, "C": 1, "D": 0}, mask

    # Largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) + 1e-6
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)
    lesion_crop = m[y:y+h, x:x+w]

    # Asymmetry
    if lesion_crop.size == 0:
        asymmetry = 0.0
    else:
        padded = np.zeros((max(lesion_crop.shape),) * 2, dtype=np.uint8)
        padded[:lesion_crop.shape[0], :lesion_crop.shape[1]] = lesion_crop
        mid = padded.shape[1] // 2
        left = padded[:, :mid]
        right = padded[:, mid:]
        right = np.fliplr(right)
        min_r = min(left.shape[0], right.shape[0])
        min_c = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:min_r, :min_c] - right[:min_r, :min_c])
        asymmetry = float(np.sum(diff) / (np.sum(padded) + 1e-6))
        asymmetry = max(0.0, min(asymmetry, 1.0))

    # Border irregularity
    border_irreg = float((perimeter ** 2) / (4 * np.pi * area + 1e-6))

    # Color clusters
    lesion_pixels = img[m == 255]
    if lesion_pixels.shape[0] >= 20:
        k = min(4, max(1, lesion_pixels.shape[0] // 500))
        try:
            km = KMeans(n_clusters=k, n_init=4, random_state=0)
            km.fit(lesion_pixels)
            color_clusters = len(np.unique(km.labels_))
        except:
            color_clusters = 1
    else:
        color_clusters = 1

    # Diameter = max dimension of bounding box
    diameter = int(max(w, h))

    return {
        "A": asymmetry,
        "B": border_irreg,
        "C": int(color_clusters),
        "D": int(diameter)
    }, mask
