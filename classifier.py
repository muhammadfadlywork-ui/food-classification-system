import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm


def resize_with_padding(img: Image.Image, target_size=(380, 380), fill_color=(0, 0, 0)):
    img = img.convert("RGB")
    target_w, target_h = target_size
    img_w, img_h = img.size

    img_ratio = img_w / img_h
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_w = target_w
        new_h = int(new_w / img_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * img_ratio)

    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized_img, (x, y))

    return canvas


class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=False)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ImageClassifier:
    """
    Dipakai sama seperti sebelumnya:
      ImageClassifier(model_path, class_names_or_json, target_size=(380,380))

    - model_path: path ke .pth (BEST EMA)
    - class_names_or_json: bisa list class names, atau path ke classes.json
    - target_size: (380,380)
    """
    def __init__(self, model_path, class_names_or_json, target_size=(380, 380), dropout=0.3, device=None):
        self.target_size = target_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # class names: boleh list langsung, atau path classes.json
        if isinstance(class_names_or_json, (list, tuple)):
            self.class_names = list(class_names_or_json)
        elif isinstance(class_names_or_json, str) and class_names_or_json.lower().endswith(".json"):
            with open(class_names_or_json, "r", encoding="utf-8") as f:
                mapping = json.load(f)  # {"0":"Asinan...", ...}
            self.class_names = [mapping[str(i)] for i in range(len(mapping))]
        else:
            raise ValueError("class_names_or_json harus list class names atau path ke classes.json")

        num_classes = len(self.class_names)

        # build model
        self.model = EfficientNetV2S(num_classes=num_classes, dropout=dropout)

        # load checkpoint .pth
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)  # handle jika langsung state_dict
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # normalize sama training kamu
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _preprocess_image(self, filepath):
        img = Image.open(filepath).convert("RGB")
        img = resize_with_padding(img, target_size=self.target_size, fill_color=(0, 0, 0))

        x = np.array(img, dtype=np.float32) / 255.0  # HWC [0..1]
        x = torch.from_numpy(x).permute(2, 0, 1)     # CHW

        x = (x - self.mean) / self.std               # normalize
        return x.unsqueeze(0)                        # (1,3,H,W)

    @torch.no_grad()
    def predict(self, filepath):
        x = self._preprocess_image(filepath).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_index = int(torch.argmax(probs).item())
        return self.class_names[pred_index]
