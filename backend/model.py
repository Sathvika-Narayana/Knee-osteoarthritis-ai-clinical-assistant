import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121

DEVICE = "cpu"
NUM_CLASSES = 5

# 🔥 Switch model here ONLY
MODEL_TYPE = "resnet"   # "resnet" or "densenet"

MODEL_PATHS = {
    "resnet": "backend/models/resnet50_fold_6.pt",
    "densenet": "backend/models/densenet_fold_4.pt"
}


# =========================
# RESNET (MATCH TRAINING)
# =========================

class MentorResNet(nn.Module):

    def __init__(self):
        super().__init__()

        base = resnet50(weights=None)

        self.backbone = nn.Sequential(*list(base.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, NUM_CLASSES)
        )

    def forward(self,x):

        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# =========================
# DENSENET (MATCH TRAINING EXACTLY)
# =========================

class MentorDenseNet(nn.Module):

    def __init__(self):
        super().__init__()

        base = densenet121(weights=None)

        self.backbone = base.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self,x):

        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# =========================
# LOAD MODEL
# =========================

def load_model():

    if MODEL_TYPE == "resnet":
        model = MentorResNet()

    elif MODEL_TYPE == "densenet":
        model = MentorDenseNet()

    else:
        raise ValueError("Invalid MODEL_TYPE")

    checkpoint = torch.load(
        MODEL_PATHS[MODEL_TYPE],
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print(f"✅ Loaded {MODEL_TYPE} model")

    return model