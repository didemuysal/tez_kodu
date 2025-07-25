import h5py
import cv2
import numpy as np
import torch
from torchvision import transforms

def prepare_image_for_model(path_to_mat_file):
    with h5py.File(path_to_mat_file, "r") as file:
        image_data = file["cjdata"]["image"][()]
        true_label_index = int(file["cjdata"]["label"][0][0]) - 1

    processed_image_data_for_model = image_data.astype(np.float32)
    processed_image_data_for_model /= processed_image_data_for_model.max()

    # This part prepares the image for visualization with Grad-CAM
    image_for_visualization = np.uint8(255 * image_data / image_data.max())
    image_for_visualization = cv2.cvtColor(image_for_visualization, cv2.COLOR_GRAY2RGB)
    image_for_visualization = cv2.resize(image_for_visualization, (224, 224))
    image_for_visualization = np.float32(image_for_visualization) / 255

    # This part applies the same transformations used during training
    # to prepare the image for input to the model
    training_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_input_tensor = training_transforms(processed_image_data_for_model)
    model_input_tensor = model_input_tensor.unsqueeze(0)

    return model_input_tensor, image_for_visualization, true_label_index