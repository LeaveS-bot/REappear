import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 反归一化
def deprocess(tensor):
    tensor = tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    tensor = tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(tensor, 0, 1)

# Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradients_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(save_activations_hook)
        target_layer.register_backward_hook(save_gradients_hook)

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cv2.resize(cam.detach().cpu().numpy(), (224, 224))
        return cam

# 图像加载与GradCAM使用示例
def main():
    img_path = 'cat.jpg'  # 使用你自己的图像路径
    image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)

    model = models.resnet50(pretrained=True)
    gradcam = GradCAM(model, model.layer4[2].conv3)

    cam = gradcam(input_tensor)
    img_np = deprocess(input_tensor)

    # 可视化
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + img_np
    overlay = overlay / overlay.max()

    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
