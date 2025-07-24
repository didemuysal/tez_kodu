import os
import zipfile

ZIP_PATH = r"C:\Users\uysal\Downloads\1512427.zip"  #the downloaded zip file from figshare link
PROJECT_FOLDER = r"C:\Users\uysal\Desktop\tez_kodu" #project folder for the dataset

def unzip_dataset(zip_path: str, project_dir: str):
    data_raw_folder = os.path.join(project_dir, "data_raw")
    os.makedirs(data_raw_folder, exist_ok=True)  #create a new folder for the raw data

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(project_dir) #extract the zip file

    for item in os.listdir(project_dir):
        if item.endswith(".zip"):  #find all zip files(4 nested dataset zip file)
            nested_zip_path = os.path.join(project_dir, item)
            with zipfile.ZipFile(nested_zip_path, 'r') as nested_zf:
                nested_zf.extractall(data_raw_folder)
            os.remove(nested_zip_path) #delete empty zips

    print("Dataset is ready")

if __name__ == "__main__":
    unzip_dataset(ZIP_PATH, PROJECT_FOLDER)
