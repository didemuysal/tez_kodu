import os

def setup_experiment_folders(command_line_args):
    learning_rate_string = f"{command_line_args.lr}"
    if command_line_args.optimizer == 'adadelta':
        learning_rate_string = "default"
        
    experiment_name = f"{command_line_args.model}_{command_line_args.strategy}_{command_line_args.optimizer}_lr-{learning_rate_string}"

    main_experiments_folder = "experiments"
    this_experiment_folder = os.path.join(main_experiments_folder, experiment_name)
    
    saved_models_folder = os.path.join(this_experiment_folder, "models")
    saved_outputs_folder = os.path.join(this_experiment_folder, "outputs")

    os.makedirs(saved_models_folder, exist_ok=True)
    os.makedirs(saved_outputs_folder, exist_ok=True)

    folder_paths = {
        "base_dir": this_experiment_folder,
        "model_dir": saved_models_folder,
        "report_dir": saved_outputs_folder,
        "experiment_name": experiment_name
    }
    
    return folder_paths