import argparse
import sys
import time
import torch
import numpy as np
import pandas as pd
import os
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


def main(command_line_args):
    experiment_hyperparameters = {
        'data_folder': 'data_raw',
        'cvind_path': 'cvind.mat',
        'batch_size': 32,
        'head_epochs': 3,
        'max_epochs': 50,
        'patience': 5,
        'seed': 42,
        'num_workers': 4
    }

    for key, value in experiment_hyperparameters.items():
        setattr(command_line_args, key, value)

    all_settings = command_line_args

    torch.manual_seed(all_settings.seed)
    np.random.seed(all_settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    folder_paths = setup_experiment_folders(all_settings)
    
    print(f"Starting: {folder_paths['experiment_name']}")
    print(f"Device: {DEVICE}")

    all_fold_data_splits = create_the_folds(all_settings.data_folder, all_settings.cvind_path)

    all_folds_metric_results = []
    total_confusion_matrix = np.zeros((3, 3), dtype=int)
    all_folds_roc_data = []

    model_for_info = create_brain_tumour_model(model_name=all_settings.model)
    model_head_string = str(model_for_info.fc)
    
    dummy_train_dataset = BrainTumourDataset(all_settings.data_folder, [], [], is_train=True)
    data_augmentation_string = str(dummy_train_dataset.transform)

    for fold_number, (train_files, train_labels, val_files, val_labels, test_files, test_labels) in enumerate(all_fold_data_splits):
        current_fold = fold_number + 1
        print(f"\n--- Starting Fold {current_fold}/5 ---")

        train_dataset = BrainTumourDataset(all_settings.data_folder, train_files, train_labels, is_train=True)
        val_dataset = BrainTumourDataset(all_settings.data_folder, val_files, val_labels, is_train=False)
        test_dataset = BrainTumourDataset(all_settings.data_folder, test_files, test_labels, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=all_settings.batch_size, shuffle=True, num_workers=all_settings.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=all_settings.batch_size, num_workers=all_settings.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=all_settings.batch_size, num_workers=all_settings.num_workers)

        model = create_brain_tumour_model(model_name=all_settings.model).to(DEVICE)
        loss_function = nn.CrossEntropyLoss()

        best_model_weights = train_fold(model, train_loader, val_loader, loss_function, all_settings, DEVICE)

        model_save_path = folder_paths['model_dir'] + f"/Fold_{current_fold}_best_model.pth"
        torch.save(best_model_weights, model_save_path)
        print(f"Model saved: {model_save_path}")

        print("\nTesting model...")
        model.load_state_dict(best_model_weights)

        test_loss, test_labels, test_predictions, test_scores = run_one_epoch(model, test_loader, loss_function, None, DEVICE)

        fold_results, roc_data = calculate_test_metrics_for_fold(test_labels, test_predictions, test_scores)
        fold_results['fold'] = current_fold
        fold_results['test_loss'] = test_loss
        all_folds_metric_results.append(fold_results)

        total_confusion_matrix += confusion_matrix(test_labels, test_predictions, labels=[0, 1, 2])
        all_folds_roc_data.append(roc_data)
        
        print(f"Fold {current_fold} Test Accuracy: {fold_results['test_accuracy']:.3%}")
        print(f"Fold {current_fold} Test Loss: {test_loss:.4f}")

    generate_and_save_summary_report(
        all_folds_metric_results,
        total_confusion_matrix,
        all_folds_roc_data,
        folder_paths,
        folder_paths['experiment_name']
    )
    
    results_df = pd.DataFrame(all_folds_metric_results)
    mean_metrics = results_df.drop(columns=['fold']).mean().to_dict()
    std_metrics = results_df.drop(columns=['fold']).std().to_dict()

    experiment_details = {
        "run_date": time.ctime(),
        "device": DEVICE,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "command_line_arguments": vars(all_settings),
        "key_hyperparameters": {
            "batch_size": all_settings.batch_size,
            "head_epochs": all_settings.head_epochs,
            "max_epochs": all_settings.max_epochs,
            "patience": all_settings.patience,
            "initial_learning_rate": all_settings.lr,
            "lr_decay_factor_stage2": 10.0,
            "dataloader_num_workers": all_settings.num_workers,
        },
        "model_architecture_head": model_head_string,
        "data_augmentation_training": data_augmentation_string,
        "final_summary_metrics_mean": mean_metrics,
        "final_summary_metrics_std": std_metrics,
        "results_csv_path": os.path.relpath(os.path.join(folder_paths['report_dir'], "all_folds_results.csv"), folder_paths['base_dir']),
        "confusion_matrix_plot_path": os.path.relpath(os.path.join(folder_paths['report_dir'], "confusion_matrix.png"), folder_paths['base_dir']),
        "roc_curve_plot_path": os.path.relpath(os.path.join(folder_paths['report_dir'], "roc_curves.png"), folder_paths['base_dir'])
    }
    
    save_experiment_details(experiment_details, folder_paths['base_dir'])
    print("Experiment finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet50'])
    parser.add_argument('--strategy', type=str, choices=['finetune', 'baseline'])
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adadelta'])
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()
    main(args)