import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedevastator/comprehensive-medical-q-a-dataset")

print("Path to dataset files:", path)