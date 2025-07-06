import gradio as gr
import torch
from model import CNN3D
from preprocess import preprocess_full_heart
import os
import shutil
import zipfile
import time




# Load trained model with best hyperparams
model = CNN3D()

CHECKPOINT_PATH = "./checkpoints/epoch=32-step=957.ckpt"
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))["state_dict"])
model.eval()

def predict(zip_file, age, gender):
    try:
        # Create temp directory for extraction
        extract_dir = "temp_extracted"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir)

        # Extract zip to temp directory
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Locate the folder (assumes only one top-level folder)
        subfolders = [f.path for f in os.scandir(extract_dir) if f.is_dir()]
        if not subfolders:
            return "No valid folder found inside ZIP", 0.0
        patient_folder = subfolders[0]

        start_total = time.time()
        
        # Preprocess
        start_preprocess = time.time()
        image = preprocess_full_heart(patient_folder)
        image_tensor = image.unsqueeze(0).unsqueeze(0)  # Shape (1,1,64,64,64)
        preprocessing_time = time.time() - start_preprocess
        
        # Metadata
        gender_numeric = 1 if gender.lower() == "male" else 0
        metadata_tensor = torch.tensor([[float(age), gender_numeric]], dtype=torch.float32)

        # Predict
        start_pred = time.time()
        with torch.no_grad():
            output = model(image_tensor, metadata_tensor)
            probability = torch.sigmoid(output).item()
        inference_time = time.time() - start_pred

        label = "Takotsubo Syndrome" if probability > 0.5 else "Normal"

        total_time = time.time() - start_total
        
        # Cleanup
        shutil.rmtree(extract_dir)

        return label, round(probability, 3), round(preprocessing_time, 2), round(inference_time, 2), round(total_time, 2)

    except Exception as e:
        return f"Error: {str(e)}", 0.0

# GUI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload DICOM Folder as ZIP", type="file"),
        gr.Number(label="Patient Age (years)", value=70),
        gr.Radio(choices=["Male", "Female"], label="Patient Gender", value="Female")
    ],
    outputs=[
        gr.Label(label="Predicted Diagnosis"),
        gr.Number(label="Probability (0–1)"),
        gr.Textbox(label="Preprocessing Time (s)"),
        gr.Textbox(label="Inference Time (s)"),
        gr.Textbox(label="Total Time (s)")
    ],
    title="Takotsubo Syndrome Detection",
    description="Upload a ZIP containing one DICOM folder (e.g., `LET_12345678.zip`). Age and gender improve model accuracy.",
    theme="default"
)

if __name__ == "__main__":
    iface.launch()