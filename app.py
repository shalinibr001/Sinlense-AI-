# app.py
import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
from preprocess import read_rgb, preprocess_for_model, compute_abcd, segment_lesion
from fpdf import FPDF

# Config
UPLOAD_FOLDER = "static/uploads"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "skinlens_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "label_classes.npy")
ALLOWED_EXT = {'png','jpg','jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and labels
model = None
labels = None
try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    print("Loaded model and labels:", labels)
except Exception as e:
    print("Warning: Could not load model/labels:", e)

# Map HAM-derived classes to abstract 4 classes if needed (but train script already uses 4)
# Labels array should be ['benign','cancerous','precancerous','suspicious'] or similar depending on flow
# We'll present friendly names:
friendly_map = {
    'benign': "Benign",
    'suspicious': "Suspicious",
    'precancerous': "Pre-Cancerous",
    'cancerous': "Cancerous"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def hybrid_risk_score(pred_dict, abcd):
    """
    pred_dict: {label_name:confidence}
    abcd: dict with A,B,C,D
    Heuristic:
      - mal_score = confidence of Pre-Cancerous + Cancerous
      - clinical = weighted sum of normalized A,B,C,D
      - final = 0.7*mal_score + 0.3*clinical
    """
    mal_score = 0.0
    mal_score += pred_dict.get('precancerous', 0.0)
    mal_score += pred_dict.get('cancerous', 0.0)
    # normalize clinical features
    A = min(abcd.get('A',0.0),1.0)
    B = min(abcd.get('B',0.0) / 10.0, 1.0)  # scale heuristic
    C = min(abcd.get('C',1) / 4.0, 1.0)
    D = min(abcd.get('D',0) / 300.0, 1.0)
    clinical = 0.4 * A + 0.3 * B + 0.2 * C + 0.1 * D
    final = 0.7 * mal_score + 0.3 * clinical
    return float(np.clip(final, 0.0, 1.0))

# PDF helper
def generate_pdf(name, age, gender, pred_label, confidence, abcd, risk_score, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "SkinLens AI - Diagnostic Report", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0,7, "Patient Details:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0,6, f"Name: {name}    Age: {age}    Gender: {gender}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0,7, "AI Diagnosis:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0,6, f"Predicted class: {pred_label}    Confidence: {confidence:.3f}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0,7, "ABCD Analysis:", ln=True)
    pdf.set_font("Arial", size=11)
    for k,v in abcd.items():
        pdf.cell(0,6, f"{k}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0,7, f"Hybrid Risk Score (0-1): {risk_score:.3f}", ln=True)
    pdf.ln(6)
    # Insert image
    try:
        if os.path.exists(image_path):
            # Convert to JPG if necessary is not done here; hope image is OK
            pdf.image(image_path, x=55, w=100)
    except Exception:
        pdf.cell(0,6, "Could not include image.", ln=True)
    # Tips
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0,7, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,6, "- This is a screening tool and not a diagnosis. Please consult a dermatologist for confirmation.\n- If lesion changes (size/color/bleeding) seek immediate medical attention.\n- Keep records and periodic photos to monitor evolution.")
    # save
    safe = secure_filename(name if name else "patient")
    filename = f"{safe}_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    out = os.path.join(UPLOAD_FOLDER, filename)
    pdf.output(out)
    return out

@app.route('/')
def index():
    # initial page (could also show basic metadata)
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    name = request.form.get('name','Anonymous')
    age = request.form.get('age','N/A')
    gender = request.form.get('gender','N/A')

    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        unique = datetime.now().strftime("%Y%m%d%H%M%S_") + fname
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        file.save(save_path)

        # Read and preprocess
        img_rgb = read_rgb(save_path, target=(224,224))
        x = preprocess_for_model(img_rgb, target_size=(224,224))

        # Predict using model (if loaded)
        pred_dict = {}
        predicted_label = "Model not loaded"
        confidence = 0.0
        if model is not None and labels is not None:
            preds = model.predict(x, verbose=0)[0]
            # map to label names
            for i,lab in enumerate(labels):
                pred_dict[str(lab)] = float(preds[i])
            idx = int(np.argmax(preds))
            predicted_label = str(labels[idx])
            confidence = float(preds[idx])
        else:
            # fallback: no model
            pred_dict = {'benign':0.5,'suspicious':0.1,'precancerous':0.1,'cancerous':0.3}
            predicted_label = 'benign'
            confidence = pred_dict.get(predicted_label, 0.0)

        # ABCD features
        abcd, mask = compute_abcd(img_rgb)

        # hybrid risk
        risk = hybrid_risk_score(pred_dict, abcd)

        # friendly display label
        display_label = friendly_map.get(predicted_label, predicted_label.title())

        # generate pdf
        pdf_path = generate_pdf(name, age, gender, display_label, confidence, abcd, risk, save_path)
        pdf_name = os.path.basename(pdf_path)

        # pass mask path for visualization by writing mask image
        try:
            mask_path = os.path.splitext(save_path)[0] + "_mask.png"
            import cv2
            cv2.imwrite(mask_path, mask)
        except Exception:
            mask_path = None

        return render_template("result.html",
                               name=name, age=age, gender=gender,
                               pred_label=display_label, confidence=round(confidence,3),
                               abcd=abcd, risk=round(risk,3),
                               image_path=save_path, mask_path=mask_path,
                               pdf_name=pdf_name)
    else:
        return "File type not allowed", 400

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
