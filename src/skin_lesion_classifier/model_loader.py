"""Loads the pretrained HAM10000 ViT from Hugging Face and wires it into an InferenceService."""

from transformers import ViTForImageClassification, ViTImageProcessor

from skin_lesion_classifier.inference import InferenceService

MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"
# The model repo ships no image-processor config, so we use the one from its base checkpoint.
BASE_PROCESSOR_ID = "google/vit-base-patch16-224-in21k"


def load_inference_service(model_id: str = MODEL_ID) -> InferenceService:
    model = ViTForImageClassification.from_pretrained(model_id)
    processor = ViTImageProcessor.from_pretrained(BASE_PROCESSOR_ID)
    return InferenceService(model=model, processor=processor)
