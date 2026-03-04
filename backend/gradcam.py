# backend/gradcam.py

import torch
import cv2
import numpy as np


def generate_gradcam(model, image_tensor, image_pil):

    gradients = []
    activations = []

    # 🔥 Detect correct target layer automatically
    if hasattr(model, "backbone"):
        target_layer = model.backbone[-1]
    else:
        raise ValueError("Model does not have backbone attribute")

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().detach().cpu().numpy()

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(
        heatmap,
        (image_pil.size[0], image_pil.size[1])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(image_pil)

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Remove hooks (VERY IMPORTANT)
    handle_forward.remove()
    handle_backward.remove()

    return superimposed