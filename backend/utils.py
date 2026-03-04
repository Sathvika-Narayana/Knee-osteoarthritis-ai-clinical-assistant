from PIL import Image
import torch
from torchvision import transforms

#  MUST match training transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess_image(image_pil):

    image_pil = image_pil.convert("RGB")

    tensor = transform(image_pil).unsqueeze(0)

    return tensor