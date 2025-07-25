import argparse
import torch
import cv2
import numpy as np
import os
import sys

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_brain_tumour_model
from class_names import class_names
from gradcam_utils import prepare_image_for_model


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a brain tumor image.")
    
    parser.add_argument('--experiment_folder', type=str, required=True, 
                        help='Name of the experiment folder, e.g., "resnet50_finetune_adam_lr-0.0001".')
    parser.add_argument('--fold_number', type=int, required=True,
                        help="The fold number of the model you  want to use (e.g., 1).")
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet18', 'resnet50'],
                        help="The model architecture, e.g., 'resnet50'.")
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the brain scan image you want to analyze (.mat file).')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_filename = f"Fold_{args.fold_number}_best_model.pth"
    path_to_model = os.path.join("experiments", args.experiment_folder, "models", model_filename)
    
    if not os.path.exists(path_to_model):
        print(f"Model file not found")
        sys.exit(1)

    print(f"Loading model: {path_to_model}")
    model = create_brain_tumour_model(model_name=args.model_type, pretrained=False).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    model_input, image_for_viz, true_label_index = prepare_image_for_model(args.image_path)
    model_input = model_input.to(device)

    target_layer = [model.layer4[-1]]
    cam_algorithm = GradCAM(model=model, target_layers=target_layer)
    
    cam_output = cam_algorithm(input_tensor=model_input)
    cam_output = cam_output[0, :]

    visualization = show_cam_on_image(image_for_viz, cam_output, use_rgb=True)
    
    output_scores = torch.softmax(model(model_input), dim=1)
    predicted_label_index = output_scores.argmax().item()
    
    predicted_class_name = class_names.get(predicted_label_index)
    true_class_name = class_names.get(true_label_index)

    output_directory = "gradcam_outputs"
    os.makedirs(output_directory, exist_ok=True)
    
    image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_filename = f"gradcam_{image_basename}_true-{true_class_name}_pred-{predicted_class_name}.png"
    output_path = os.path.join(output_directory, output_filename)

    cv2.imwrite(output_path, visualization)
    print(f"Grad-CAM image saved to: {output_path}")


if __name__ == '__main__':
    main()