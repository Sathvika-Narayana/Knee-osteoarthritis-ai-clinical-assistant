from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
import torch
import cv2

from backend.model import load_model, MODEL_TYPE
from backend.utils import preprocess_image
from backend.gradcam import generate_gradcam
from backend.explanation import generate_explanation

app = FastAPI()

# 🔥 Load model once during startup (FAST + stable)
model = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # ✅ Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ Preprocess image
        image_tensor = preprocess_image(image)

        # IMPORTANT → ensure tensor only (avoid conv2d tuple error)
        if isinstance(image_tensor, tuple):
            image_tensor = image_tensor[0]

        # ✅ Prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, 1).item()

        # ✅ GradCAM
        gradcam_img = generate_gradcam(model, image_tensor, image)

        # convert image → base64
        _, buffer = cv2.imencode(".jpg", gradcam_img)
        gradcam_base64 = base64.b64encode(buffer).decode()

        # ✅ LLM explanation (Ollama)
        explanation = generate_explanation(f"KL{pred}")

        return {
            "prediction": f"KL{pred}",
            "gradcam_image": f"data:image/jpeg;base64,{gradcam_base64}",
            "explanation": explanation
        }

    except Exception as e:

        print("Backend Processing Error:", e)

        return {
            "error": str(e)
        }