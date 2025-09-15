import torch
from src.genetic import Hyperparameters
from src.predict import load_model, classify_images

if __name__ == "__main__":
    model_path = "ga_optimized_model.pth"
    image_dir = "test"
    output_file = "results.txt"

    hyperparams = Hyperparameters(
        width_mult=1.8574, learning_rate=0.077489, batch_size=64,
        dropout_rate= 0.1426, weight_decay=0.000159, momentum=0.8017,
        conv_channels=[16, 32, 64, 128]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, hyperparams, device)
    classify_images(model, image_dir, output_file, device)
