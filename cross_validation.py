import os
import h5py
import numpy as np

def create_the_folds(folder_with_data, crossval_file): #which one belongs to which fold
    with h5py.File(crossval_file, "r") as f:
        patient_fold_map = f["cvind"][()].flatten()

    def get_number(name_of_file):  #helper func to get the numberic file name
        return int(os.path.splitext(name_of_file)[0])

    just_the_mat_files = [] #sort them numerically
    for a_file in os.listdir(folder_with_data):
        if a_file.endswith(".mat"):
            just_the_mat_files.append(a_file)

    sorted_filenames = sorted(just_the_mat_files, key=get_number)
    sorted_filenames = np.array(sorted_filenames)

    patient_labels = [] #get labels
    for a_file in sorted_filenames:
        full_path_to_file = os.path.join(folder_with_data, a_file)
        with h5py.File(full_path_to_file, "r") as f:
            patient_labels.append(int(f["cjdata"]["label"][0][0]))
    patient_labels = np.array(patient_labels)

    list_of_all_folds = [] #loop 5 times to creatte train/val/test splits
    for current_fold_number in range(1, 6):
        test_set_id = current_fold_number
        val_set_id = (current_fold_number % 5) + 1  #assing the fold for test and val
        ids_for_training = [] 
        for a_fold_id in range(1, 6): #rest is for training
            if a_fold_id != test_set_id and a_fold_id != val_set_id:
                ids_for_training.append(a_fold_id)

        is_test_sample = (patient_fold_map == test_set_id)
        is_val_sample = (patient_fold_map == val_set_id)
        
        is_train_sample_list = [] #select the samples for each set
        for fold_id in patient_fold_map:
            is_train_sample_list.append(fold_id in ids_for_training)
        is_train_sample = np.array(is_train_sample_list)

        this_fold_data = ( #group the data for the corresponding fold
            sorted_filenames[is_train_sample].tolist(), patient_labels[is_train_sample].tolist(),
            sorted_filenames[is_val_sample].tolist(), patient_labels[is_val_sample].tolist(),
            sorted_filenames[is_test_sample].tolist(), patient_labels[is_test_sample].tolist()
        )
        list_of_all_folds.append(this_fold_data)

    return list_of_all_folds