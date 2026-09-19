"""ViT inference service: image in, HAM10000 diagnosis with confidence out."""

from dataclasses import dataclass

import torch
from PIL import Image

# HAM10000 short codes, keyed by the label names the pretrained model was trained with.
MODEL_LABEL_TO_CODE = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanocytic_Nevi": "nv",
    "melanoma": "mel",
    "vascular_lesions": "vasc",
}
HAM10000_LABELS = frozenset(MODEL_LABEL_TO_CODE.values())


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class InferenceService:
    def __init__(self, model, processor) -> None:
        self._model = model.eval()
        self._processor = processor

    def predict(self, image: Image.Image) -> Prediction:
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        scores = torch.softmax(logits, dim=-1)[0].tolist()
        probabilities = {
            MODEL_LABEL_TO_CODE[self._model.config.id2label[index]]: score
            for index, score in enumerate(scores)
        }
        label = max(probabilities, key=probabilities.get)
        return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)
