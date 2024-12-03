# biotrack, CC-BY-NC license
# Filename: biotrack/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from typing import List

from biotrack.logger import info, err

from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch.nn.functional as F

class ViTWrapper:
    #DEFAULT_MODEL_NAME = "/Users/dcline/Dropbox/code/biotrack/models/i2MAP-vit-b-16"
    DEFAULT_MODEL_NAME = "/mnt/DeepSea-AI/models/m3midwater-vit-b-16/"
    # DEFAULT_MODEL_NAME = "/mnt/DeepSea-AI/models/m3midwater-vit-b-16/"


    def __init__(self, device: str = "cpu", batch_size: int = 32, model_name: str = DEFAULT_MODEL_NAME, device_id: int = 0):
        self.batch_size = batch_size
        self.name = model_name
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        if not Path(model_name).exists():
            err(f"Model {model_name} does not exist")
            raise FileNotFoundError(f"Model {model_name} does not exist")

        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def process_images(self, image_paths: list):
        info(f"Processing {len(image_paths)} images with {self.model_name}")

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            embeddings = self.model.base_model(**inputs)
            batch_embeddings = embeddings.last_hidden_state[:, 0, :].cpu().numpy()

        # Get the top 3 classes and scores
        top_scores, top_classes = torch.topk(logits, 3)
        top_classes = top_classes.cpu().numpy()
        top_scores = F.softmax(top_scores, dim=-1).cpu().numpy()
        predicted_classes = [[self.model.config.id2label[class_idx] for class_idx in class_list] for class_list in top_classes]
        predicted_scores = [[score for score in score_list] for score_list in top_scores]

        return batch_embeddings, predicted_classes, predicted_scores