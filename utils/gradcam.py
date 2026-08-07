"""
gradcam.py
----------
Grad-CAM implementation used to visualize the spatial regions of a keyframe
that most influence the DD-RCNN's plot-classification decision (manuscript
Figure 5: Grad-CAM heatmaps for 5 input samples).

Standard Grad-CAM formulation:
    alpha_k = (1 / Z) * sum_{i,j} dY^c / dA^k_ij
    L^c_GradCAM = ReLU( sum_k alpha_k * A^k )
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Generic Grad-CAM wrapper: hooks into a target convolutional layer of a
    model, captures activations and gradients during a forward/backward
    pass, and produces a class-discriminative localization heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor.
            class_idx: target class index; if None, uses the predicted class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1], resized to input resolution.
            class_idx: the class index used for the CAM.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)          # expects logits (1, num_classes)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]               # (C, H, W)
        activations = self.activations[0]            # (C, H, W)

        alpha = gradients.mean(dim=(1, 2))            # alpha_k, global-average-pooled gradients
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for k, a_k in enumerate(alpha):
            cam += a_k * activations[k]

        cam = F.relu(cam)                             # ReLU(sum_k alpha_k * A^k)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()
        h, w = input_tensor.shape[-2:]
        heatmap = cv2.resize(cam_np, (w, h))
        return heatmap, class_idx


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto the original image for visualization
    (as in Figure 5).

    Args:
        image: HxWx3 BGR uint8 image.
        heatmap: HxW float array in [0, 1].
        alpha: blending factor for the heatmap overlay.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, color_map, alpha, 0)
    return overlay


if __name__ == "__main__":
    # Minimal smoke test with a small dummy CNN classifier.
    class DummyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
            )
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(32, 5)

        def forward(self, x):
            feat = self.features(x)
            pooled = self.pool(feat).flatten(1)
            return self.fc(pooled)

    model = DummyClassifier()
    target_layer = model.features[-2]
    cam_tool = GradCAM(model, target_layer)

    dummy_input = torch.randn(1, 3, 64, 64, requires_grad=True)
    heatmap, cls_idx = cam_tool.generate(dummy_input)
    print("Grad-CAM heatmap shape:", heatmap.shape, "predicted class:", cls_idx)
