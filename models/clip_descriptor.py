"""
clip_descriptor.py
-------------------
CLIP-based semantic alignment module used to retrieve the most contextually
relevant caption for each detected graphical region (manuscript Section:
CLIP-descriptor module, Eq. 12):

    sim(I, T_j) = (E_I . E_Tj) / (||E_I|| * ||E_Tj||)

The text segment with the highest cosine similarity to the cropped region's
image embedding is selected as the caption, i.e. retrieval-based captioning
rather than generative captioning.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class CLIPDescriptor(nn.Module):
    """
    Wraps a pretrained CLIP model to (a) embed cropped graphical regions,
    (b) embed OCR-extracted text candidates, and (c) rank text candidates by
    cosine similarity to select the best caption for each region.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("The `transformers` package is required for CLIPDescriptor. "
                               "Install with `pip install transformers`.")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        """Encode a list of PIL images (cropped graphical regions) into
        L2-normalized CLIP image embeddings E_I."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        embeds = self.model.get_image_features(**inputs)
        return F.normalize(embeds, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of OCR/text candidates into L2-normalized CLIP text
        embeddings E_T."""
        inputs = self.processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
        embeds = self.model.get_text_features(**inputs)
        return F.normalize(embeds, dim=-1)

    def cosine_similarity(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix between image and text embeddings (Eq. 12)."""
        return image_embeds @ text_embeds.t()

    def select_caption(self, image, text_candidates: List[str]) -> Tuple[str, float]:
        """
        Retrieve the best-matching caption for a single cropped graphical
        region from a list of OCR text candidates.

        Args:
            image: a single PIL image (cropped ROI).
            text_candidates: list of candidate text segments extracted via OCR.

        Returns:
            (best_caption, similarity_score)
        """
        if len(text_candidates) == 0:
            return "", 0.0

        image_embed = self.encode_images([image])              # (1, D)
        text_embeds = self.encode_texts(text_candidates)        # (N, D)
        sims = self.cosine_similarity(image_embed, text_embeds).squeeze(0)  # (N,)

        best_idx = torch.argmax(sims).item()
        return text_candidates[best_idx], sims[best_idx].item()

    def batch_select_captions(self, images: List, text_candidates_per_image: List[List[str]]):
        """Vectorized version of `select_caption` for a batch of ROI crops,
        each with its own list of candidate text segments."""
        results = []
        for image, candidates in zip(images, text_candidates_per_image):
            results.append(self.select_caption(image, candidates))
        return results


def clip_contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                           temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE-style contrastive loss used to align image and caption
    embeddings during training (manuscript Section: Loss function -- CLIP
    caption-alignment loss L_cap). Correct image-caption pairs are assumed to
    lie on the diagonal of the similarity matrix.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


if __name__ == "__main__":
    # Minimal smoke test of the contrastive loss (does not require a
    # downloaded CLIP checkpoint).
    dummy_img_embeds = torch.randn(4, 512)
    dummy_text_embeds = torch.randn(4, 512)
    loss = clip_contrastive_loss(dummy_img_embeds, dummy_text_embeds)
    print("CLIP contrastive loss:", loss.item())
