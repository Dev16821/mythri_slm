from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path=".", repo_id="Deviprasd67tugyjhbm/mythri_slm", repo_type="model")
