# Dissertation Mammogram Classification
Dissertation Project: An interpretable deep learning pipeline for binary classification of mammograms, using CNNs, Grad-CAM visualisations, and transfer learning. Includes a local web interface for model testing and explanation.


## Project Structure

```shell
mammogram-ai-project/
│
├── data_preprocessing/      # Preprocessing and dataset preparation
│   ├── dicom_to_png_conversion.ipynb
│   ├── sort_mammogram_data.ipynb
│   ├── sam_preprocessing_images.ipynb
│   ├── custom_dataset.ipynb
│   └── CLAHE_histogram_equalization.ipynb
│
├── model_training/          # Kaggle-compatible training notebooks
│   ├── train_densenet.ipynb
│   ├── train_efficientnet.ipynb
│   └── train_mobilenetv3.ipynb
│
├── model_testing/           # Model evaluation notebooks
│   └── test_models.ipynb
│
├── gradcam_analysis/        # Grad-CAM visualisation
│   ├── compare_model_heatmaps.ipynb
│   └── explore_gradcam.ipynb
│
├── webapp/                  # Flask web interface
│   ├── static/              # Static assets (images, models, results)
│   ├── templates/           # HTML templates
│   └── app.py               # Flask backend script
│
├── requirements.txt         # Project dependencies
├── README.md                # Documentation overview
├── .gitignore               # Ignored files (datasets, temp files, etc.)
└── .gitattributes           # Git LFS configuration for model weights
```

## Setting up the Project

### Clone the repository
```shell
git clone https://github.com/giuliabrown/dissertation-mammogram-classification.git
cd mammogram-ai-project
```

### Set up the virtual environment
```shell
# Install virtualenv package
pip install virtualenv

# In the directory containing the source code, create the virtual environment
python3.11 -m venv env

# Activate virtual environment and install requirements.txt

# For MacOS/Linux:
source env/bin/activate

# You should see (env) before the command prompt
pip install -r requirements.txt
```


## Data Setup

### Downloading data
- Preprocessed PNG data: Download the preprocessed "Data png cropped" folder from the provided OneDrive link. Place it into the following location:
`mammogram-ai-project/Data`
- Original DICOM files: For some preprocessing notebooks (e.g., DICOM to PNG), you need to download the original mammogram dataset from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).


## Running the notebooks
- Model Training: Modify paths in notebooks inside `model_training/` (originally configured for Kaggle). Ensure your data is placed correctly, then run these notebooks locally or upload them to Kaggle.
- Model Testing and Grad-CAM: Ensure your models (`.pth` files) are available in `webapp/static/models`. Run notebooks in `model_testing/` and `gradcam_analysis/` for evaluation and visual explanations.


### Running the webapp

Activate the virtual environment:

```shell
source env/bin/activate
```

Navigate to the webapp folder `mammogram-ai-project/webapp/`, and start the webapp as shown:

```shell
cd mammogram-ai-project/webapp
python3.11 app.py
```

Open a browser and go to http://127.0.0.1:5000.


## Model Weights (.pth files)

Models (`.pth`) are stored using Git Large File Storage (Git LFS). When cloning the repository, ensure you have Git LFS installed:

```shell
git lfs install
git clone https://github.com/giuliabrown/dissertation-mammogram-classification.git
```