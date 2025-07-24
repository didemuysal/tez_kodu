import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (auc, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)
from sklearn.preprocessing import label_binarize

def calculate_test_metrics_for_fold(true_labels, predicted_labels, model_scores):
    number_of_classes = len(np.unique(true_labels))
    class_names = ['meningioma', 'glioma', 'pituitary']
    
    accuracy = (predicted_labels == true_labels).mean()
    
    precision, recall, f1_score, class_sample_counts = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, labels=list(range(number_of_classes)), zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, macro_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', labels=list(range(number_of_classes)), zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', labels=list(range(number_of_classes)), zero_division=0
    )
    
    fold_results_dict = {
        'test_accuracy': accuracy,
        'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f1': macro_f1,
        'weighted_precision': weighted_precision, 'weighted_recall': weighted_recall, 'weighted_f1': weighted_f1,
    }

    one_hot_encoded_labels = label_binarize(true_labels, classes=list(range(number_of_classes)))
    roc_curve_data_for_fold = []

    for class_index, class_name in enumerate(class_names):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            one_hot_encoded_labels[:, class_index], model_scores[:, class_index]
        )
        roc_area_under_curve = auc(false_positive_rate, true_positive_rate)
        
        fold_results_dict[f"{class_name}_precision"] = precision[class_index]
        fold_results_dict[f"{class_name}_recall"] = recall[class_index]
        fold_results_dict[f"{class_name}_f1"] = f1_score[class_index]
        fold_results_dict[f"{class_name}_auc"] = roc_area_under_curve
        
        roc_curve_data_for_fold.append((false_positive_rate, true_positive_rate, class_name, roc_area_under_curve))

    return fold_results_dict, roc_curve_data_for_fold


def generate_and_save_summary_report(all_fold_results, total_confusion_matrix, all_roc_data, folder_paths, experiment_name):
    results_dataframe = pd.DataFrame(all_fold_results)
    
    mean_metrics = results_dataframe.drop(columns=['fold']).mean()
    std_dev_metrics = results_dataframe.drop(columns=['fold']).std()

    summary_df = pd.DataFrame({'Mean': mean_metrics, 'Std Dev': std_dev_metrics})
    print("\n--- Cross-Validation Complete ---")
    print("Mean and Std Dev of Performance Metrics Across 5 Folds:")
    print(summary_df)

    results_with_summary = results_dataframe.copy()
    results_with_summary.loc['Mean'] = mean_metrics
    results_with_summary.loc['Std Dev'] = std_dev_metrics
    results_csv_path = os.path.join(folder_paths['report_dir'], "all_folds_results.csv")
    results_with_summary.to_csv(results_csv_path)
    print(f"\nSaved detailed results to {results_csv_path}")

    confusion_matrix_plot_path = os.path.join(folder_paths['report_dir'], "confusion_matrix.png")
    
    row_sums = total_confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_confusion_matrix = total_confusion_matrix.astype('float') / row_sums
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_confusion_matrix, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Meningioma', 'Glioma', 'Pituitary'], 
                yticklabels=['Meningioma', 'Glioma', 'Pituitary'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix\n{experiment_name}')
    plt.savefig(confusion_matrix_plot_path)
    plt.close()
    print(f"Saved confusion matrix to {confusion_matrix_plot_path}")

    roc_curve_plot_path = os.path.join(folder_paths['report_dir'], "roc_curves.png")
    plt.figure(figsize=(8, 6))
    
    mean_false_positive_rate = np.linspace(0, 1, 100)
    class_names = ['Meningioma', 'Glioma', 'Pituitary']
    
    for class_index, class_name in enumerate(class_names):
        true_positive_rates_for_class = []
        aucs_for_class = []
        
        for single_fold_roc_data in all_roc_data:
            for fpr, tpr, curve_name, roc_auc in single_fold_roc_data:
                if curve_name.lower() == class_name.lower():
                    true_positive_rates_for_class.append(np.interp(mean_false_positive_rate, fpr, tpr))
                    aucs_for_class.append(roc_auc)
        
        mean_tpr = np.mean(true_positive_rates_for_class, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs_for_class)
        std_auc = np.std(aucs_for_class)
        
        plt.plot(mean_false_positive_rate, mean_tpr, lw=2, 
                 label=f'{class_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')


    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {experiment_name}')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_plot_path)
    plt.close()
    print(f"Saved ROC curve to {roc_curve_plot_path}")


def save_experiment_details(details_to_save, path_to_save_at):
    details_file_path = os.path.join(path_to_save_at, "experiment_details.json")
    with open(details_file_path, 'w') as json_file:
        json.dump(details_to_save, json_file, indent=4)
    print(f"Experiment details saved to: {details_file_path}")