# biotrack, CC-BY-NC license
# Filename: biotrack/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Callable, Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from biotrack.batch_utils import percent_coverage, reshape_transform
from biotrack.delme import HuggingfaceToTensorModelWrapper
from biotrack.logger import info, err, debug
from biotrack.track import Track

from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
import torch.nn.functional as F


class ViTWrapper:
    # DEFAULT_MODEL_NAME = "/Users/dcline/Dropbox/code/biotrack/models/i2MAP-vit-b-16"
    # DEFAULT_MODEL_NAME = "/mnt/DeepSea-AI/models/m3midwater-vit-b-16/"
    # DEFAULT_MODEL_NAME = "/Volumes/DeepSea-AI/models/m3midwater-vit-b-16/"
    # DEFAULT_MODEL_NAME = "/mnt/DeepSea-AI/models/i2MAP-vit-b-16/"
    # DEFAULT_MODEL_NAME = "/Users/dcline/Dropbox/code/biotrack/models/i2MAP-vit-b-16"
    DEFAULT_MODEL_NAME = "/Users/dcline/Dropbox/code/biotrack/models/m3midwater-vit-b-16/"


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
        self.target_layer_gradcam = self.model.vit.encoder.layer[-2].output

    def category_name_to_index(self, category_name):
        name_to_index = dict((v, k) for k, v in self.model.config.id2label.items())
        return name_to_index[category_name]

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def process_images(self, image_paths: list, min_coverage: float = 10.0) -> tuple:
        info(f"Processing {len(image_paths)} images with {self.model_name}")

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)["pixel_values"]

        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits

        # Get the top 10 classes and scores
        top_scores, top_classes = torch.topk(logits, 10)
        top_classes = top_classes.cpu().numpy()
        top_scores = F.softmax(top_scores, dim=-1).cpu().numpy()

        coverages = []
        keypoints = []
        for i, image_path in enumerate(image_paths):
            best_coverage = 0.
            best_label = -1
            best_score = 0.
            best_keypoints = []
            for j, data in enumerate(zip(top_classes[i], top_scores[i])):
                label, score = data
                category_name = self.model.config.id2label[label]
                image = Image.open(image_path)
                input_tensor = transforms.ToTensor()(image)
                kp, coverage = get_gcam_keypoints(model=self.model,
                                        num_keypoints=Track.NUM_KP,
                                        target_layer=self.target_layer_gradcam,
                                        input_tensor=input_tensor.to(self.device),
                                        targets_for_gradcam=[ClassifierOutputTarget(self.category_name_to_index(category_name))],
                                        reshape_transform=reshape_transform)
                if coverage > best_coverage and coverage > min_coverage:
                    if best_label == -1:
                        best_label = label
                    top_classes[i][0] = label # Update the top 2 classes and scores
                    top_scores[i][0] = score
                    top_classes[i][1] = best_label
                    top_scores[i][1] = best_score
                    best_label = label
                    best_score = score
                    best_coverage = coverage
                    best_keypoints = kp
                    debug(f"Found best keypoints: {best_keypoints} for {category_name} with coverage {best_coverage} for {image_path}")
                    if best_coverage > 20:
                        break

            keypoints.append(best_keypoints)
            coverages.append(best_coverage)

            if len(best_keypoints) == 0:
                err(f"No keypoints found for {image_path}")
                continue

        with torch.no_grad():
            embeddings = self.model.base_model(inputs)
            batch_embeddings = embeddings.last_hidden_state[:, 0, :].cpu().numpy()

        predicted_classes = [[self.model.config.id2label[class_idx] for class_idx in class_list] for class_list in top_classes]
        predicted_scores = [[score for score in score_list] for score_list in top_scores]

        return batch_embeddings, predicted_classes, predicted_scores, keypoints, coverages


def get_gcam_keypoints(model: torch.nn.Module,
                       target_layer: torch.nn.Module,
                       targets_for_gradcam: List[Callable],
                       reshape_transform: Optional[Callable],
                       input_tensor: torch.nn.Module,
                       num_keypoints: int = 3,
                       display: bool = False,
                       method: Callable = GradCAM):

    with (method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam):

        coverage = 0.
        keypoints = []
        repeated_tensor = input_tensor.unsqueeze(0)
        batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)

        # Get the blob for the input tensor using binary thresholding of background mean
        input_tensor = input_tensor.cpu().detach().numpy()

        input_tensor = np.transpose(input_tensor, (1, 2, 0))
        img_color = np.uint8(input_tensor * 255)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, img_thres = cv2.threshold(img_color, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_raw, _ = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if display:
            cv2.imshow("Frame", img_thres)
            cv2.waitKey(-1)

        # Add the keypoints to the results for each contour up to num_keypoints
        for grayscale_cam in batch_results:
            grayscale_cam = np.uint8(255 * grayscale_cam)
            _, grayscale_cam = cv2.threshold(grayscale_cam, 150, 255, cv2.THRESH_BINARY)

            if display:
                cv2.imshow(f"Grad Bin Frame", grayscale_cam)
                cv2.waitKey(-1)

            contours_gcam, _ = cv2.findContours(grayscale_cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours_gcam) == 0:
                continue  # No contours found

            coverage = percent_coverage(grayscale_cam, img_thres)

            if display:
                cv2.imshow(f"GradCam Frame {coverage}", grayscale_cam)
                cv2.waitKey(-1)

            # If the coverage is 0, this is not a good activation
            if coverage == 0:
                contours_gcam = contours_raw

            contour_keypoints = []
            for contour in contours_gcam:
                # Get the centroid of the contour using moments
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Disregard keypoints in the corners - they are likely artifacts of the GradCAM
                if cX < 10 or cX > grayscale_cam.shape[1] - 10 or cY < 10 or cY > grayscale_cam.shape[
                    0] - 10:
                    continue
                contour_keypoints.append([cX, cY])
                if len(contour_keypoints) >= num_keypoints:
                    break

            # If there are less than num_keypoints, add keypoints around the center of the image
            if len(contour_keypoints) < num_keypoints:
                debug(f"{len(contour_keypoints)} less than required {num_keypoints} keypoints; adding {num_keypoints - len(keypoints)} keypoints around the center of the image")
                for i in range(num_keypoints - len(contour_keypoints)):
                    offset = 10 * i
                    contour_keypoints.append([grayscale_cam.shape[1] // 2 + offset, grayscale_cam.shape[0] // 2 + offset])

            keypoints.append(contour_keypoints)
        return keypoints, coverage


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits
