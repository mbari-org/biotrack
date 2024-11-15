# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.

from PIL import Image
import numpy as np
import torch
from typing import List
import open_clip
from open_clip import tokenizer

from biotrack.logger import info, err


class ViTWrapper:
    MODEL_NAME = "ViT-B-16"
    PRETRAINED = "openai"
    VECTOR_DIMENSIONS = 512  # 512 for image, 512 for text

    def __init__(self, device: str = "cpu", batch_size: int = 8):
        self.batch_size = batch_size
        self.model, _, self.process = open_clip.create_model_and_transforms(self.MODEL_NAME, pretrained=self.PRETRAINED)
        self.tokenizer = tokenizer

        # Load the model and processor
        if "cuda" in device and torch.cuda.is_available():
            self.device = "cuda"
            self.model.to("cuda")
        else:
            self.device = "cpu"

    @property
    def model_name(self) -> str:
        return self.model.config._name_or_path


def compute_embedding_vits(vit_wrapper: ViTWrapper, images: List[str], text: List[str]):
    """
    Compute the embedding for the given images
    :param vit_wrapper: Wrapper for the ViT model
    :param images: List of image filenames
    :param text:  List of text strings to embed
    :param device: Device to use for the computation (cpu or cuda:0, cuda:1, etc.)
    :return: List of embeddings in the same order as the input images
    """
    batch_size = 8

    # Batch process the images
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

    # Store the embeddings in a list
    all_embeddings = []
    for batch in batches:
        try:
            images = [Image.open(filename).convert("RGB") for filename in batch]
            inputs = [vit_wrapper.process(image) for image in images]
            image_input = torch.tensor(np.stack(inputs))
            text_tokens = vit_wrapper.tokenizer.tokenize(["This is " + t for t in text])

            # Move the inputs to the device
            image_input = image_input.to(vit_wrapper.device)
            text_tokens = text_tokens.to(vit_wrapper.device)

            with torch.no_grad():
                image_features = vit_wrapper.model.encode_image(image_input).float()
                # text_features = vit_wrapper.model.encode_text(text_tokens).float()

            # Concatenate the image and text features
            # batch_embeddings = torch.cat([image_features, text_features], dim=1).cpu().numpy()
            batch_embeddings = image_features.cpu().numpy()

            # Save the embeddings
            for emb, filename in zip(batch_embeddings, batch):
                emb = emb.astype(np.float32)
                all_embeddings.append(emb)
        except Exception as e:
            err(f"Error processing {batch}: {e}")
            raise e

    # Return the embeddings
    return all_embeddings
