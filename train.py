import argparse
import sys
import time
import torch
import numpy as np
import pandas as pd
import os
import yaml # This import is fine, it's used for config.yaml loading
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from folders_create import setup_experiment_folders
from model_trainer import train_fold, run_one_epoch
from performance_reporter import (calculate_test_metrics_for_fold,
                                  generate_and_save_summary_report,
                                  save_experiment_details)
from data_loader import BrainTumourDataset
from model import create_brain_tumour_model
from cross_validation import create_the_folds
from class_names import class_names # Assuming this class_names.py file exists and is correct

def main(command_line_args):
    
    try:
        config_file_path = "config.yaml"
        with open(config_file_path, 'r') as file:
            yaml_loaded_settings = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure it exists in the script's directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        sys.exit(1)

    # 2. Merge YAML settings with command-line arguments.
    # YAML settings provide the base defaults.
    # Command-line arguments explicitly provided by the user will override YAML.
    all_settings_dict = {**yaml_loaded_settings}
    for key, value in vars(command_line_args).items():
        if value is not None: # If the argument was actually specified on the command line
            all_settings_dict[key] = value

    # Convert the combined dictionary to a Namespace object for easy access
    settings = argparse.Namespace(**all_settings_dict)

    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    folders = setup_experiment_folders(settings)
    print(f"Starting Experiment: {folders['experiment_name']}")
    print(f"Running on device: {device}")

    # This 'reverted' logic expects cvind.mat in the CWD, as per your preference.
    all_the_folds = create_the_folds(settings.data_folder, settings.cvind_path)

    # Correct initialization of result accumulators (these were previously noted issues)
    results_from_all_folds = []
    # Use class_names for consistent confusion matrix size
    total_confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    all_roc_data = []

    model_for_info = create_brain_tumour_model(model_name=settings.model)
    model_head_string = str(model_for_info.fc)
    
    # Corrected dummy dataset call (this was previously noted issue)
    dummy_train_dataset = BrainTumourDataset(settings.data_folder, [], [], is_train=True)
    data_augmentation_string = str(dummy_train_dataset.transform)

    for fold_index, (train_files, train_labels, val_files, val_labels, test_files, test_labels_original) in enumerate(all_the_folds):
        fold_number = fold_index + 1
        print(f"\n--- Starting Fold {fold_number}/5 ---")

        train_dataset = BrainTumourDataset(settings.data_folder, train_files, train_labels, is_train=True)
        validation_dataset = BrainTumourDataset(settings.data_folder, val_files, val_labels, is_train=False)
        test_dataset = BrainTumourDataset(settings.data_folder, test_files, test_labels_original, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=settings.batch_size, num_workers=settings.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, num_workers=settings.num_workers)

        model = create_brain_tumour_model(model_name=settings.model).to(device)
        loss_function = nn.CrossEntropyLoss()
        
        best_model_weights = train_fold(model, train_loader, validation_loader, loss_function, settings, device)
        
        model_save_path = os.path.join(folders['model_dir'], f"Fold_{fold_number}_best_model.pth")
        torch.save(best_model_weights, model_save_path)
        print(f"Best model for Fold {fold_number} saved.")

        print(f"Testing best model from Fold {fold_number}...")
        model.load_state_dict(best_model_weights)
        test_loss, returned_test_labels, test_predictions, test_scores = run_one_epoch(model, test_loader, loss_function, None, device)

        # Using test_labels_original for metrics calculating (this was previously noted issue)
        fold_results, roc_data = calculate_test_metrics_for_fold(test_labels_original, test_predictions, test_scores)
        fold_results['fold'] = fold_number
        fold_results['test_loss'] = test_loss
        results_from_all_folds.append(fold_results)

        # Using sorted class_names.keys() for consistency (this was previously noted issue)
        total_confusion_matrix += confusion_matrix(test_labels_original, test_predictions, labels=sorted(class_names.keys()))
        all_roc_data.append(roc_data)
        print(f"Fold {fold_number} Test Accuracy: {fold_results['test_accuracy']:.3%}")

    generate_and_save_summary_report(
        results_from_all_folds, total_confusion_matrix, all_roc_data,
        folders, folders['experiment_name']
    )

    # Note: model_for_info and dummy_train_dataset are defined outside the loop
    # to avoid redundant creation on every fold. This is good as is.
    model_for_info = create_brain_tumour_model(model_name=settings.model)
    model_head_string = str(model_for_info.fc)
    # data_augmentation_string is already correctly derived from dummy_train_dataset

    results_dataframe = pd.DataFrame(results_from_all_folds)
    mean_metrics = results_dataframe.drop(columns=['fold']).mean().to_dict()
    std_metrics = results_dataframe.drop(columns=['fold']).std().to_dict()

    experiment_details = {
        "run_date": time.ctime(),
        "device": device,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        # Fixed: Corrected cuda_version access based on previous feedback
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A", 
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        # Store what was passed on the CLI (command_line_args) and the effective combined settings (settings)
        "command_line_input_args": vars(command_line_args),
        "effective_experiment_settings": vars(settings),
        # Removed 'key_hyperparameters' as 'effective_experiment_settings' makes it redundant and more complete
        "model_architecture_head": model_head_string,
        "training_data_augmentation": data_augmentation_string,
        "final_summary_metrics_mean": mean_metrics,
        "final_summary_metrics_std": std_metrics,
        "results_csv_path": os.path.relpath(os.path.join(folders['report_dir'], "all_folds_results.csv"), folders['base_dir']),
        "confusion_matrix_plot_path": os.path.relpath(os.path.join(folders['report_dir'], "confusion_matrix.png"), folders['base_dir']),
        "roc_curve_plot_path": os.path.relpath(os.path.join(folders['report_dir'], "roc_curves.png"), folders['base_dir'])
    }
    
    save_experiment_details(experiment_details, folders['base_dir'])
    print("Experiment finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a brain tumor classification experiment.")

    # Core identification arguments (REQUIRED - must be specified on the command line)
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet50'],
                        help="Model architecture for training and evaluation (e.g., resnet50).")
    parser.add_argument('--strategy', type=str, required=True, choices=['finetune', 'baseline'],
                        help="Transfer learning strategy: 'finetune' (two-stage fine-tuning) or 'baseline' (fixed feature extraction).")
    parser.add_argument('--optimizer', type=str, required=True, choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta'],
                        help="Optimizer to use for network training (e.g., adam).")
    parser.add_argument('--lr', type=float, required=True,
                        help="Initial learning rate for the optimizer (e.g., 0.001).")

    # Optional hyperparameters (Defaults sourced from config.yaml).
    # --data_folder and --cvind_path are included here as they are also part of config.yaml
    # and might need to be overridden.
    parser.add_argument('--data_folder', type=str,
                        help='Overrides "data_folder" from config.yaml (e.g., data_raw).')
    parser.add_argument('--cvind_path', type=str,
                        help='Overrides "cvind_path" (filename) from config.yaml (e.g., cvind.mat).')
    parser.add_argument('--batch_size', type=int,
                        help='Overrides "batch_size" from config.yaml (e.g., 32).')
    parser.add_argument('--head_epochs', type=int,
                        help='Overrides "head_epochs" from config.yaml (e.g., 3).')
    parser.add_argument('--max_epochs', type=int,
                        help='Overrides "max_epochs" from config.yaml (e.g., 50).')
    parser.add_argument('--patience', type=int,
                        help='Overrides "patience" from config.yaml (e.g., 5).')
    parser.add_argument('--seed', type=int,
                        help='Overrides "seed" from config.yaml (e.g., 42).')
    parser.add_argument('--num_workers', type=int,
                        help='Overrides "num_workers" from config.yaml (e.g., 4).')
    
    args = parser.parse_args()
    main(args)