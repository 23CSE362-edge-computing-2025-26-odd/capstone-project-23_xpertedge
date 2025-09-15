import os
import torch
from torchvision import transforms
from PIL import Image
from src.models import CNN

def load_model(model_path, hyperparams, device):
    model = CNN(num_classes=62, hyperparams=hyperparams).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def classify_images(model, image_dir, output_file, device):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    results = []
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
        results.append(f"{img_name},{pred}")

    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Results written to {output_file}")
