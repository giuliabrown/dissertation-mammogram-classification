"""
Flask web application for mammogram classification and visualisation.

This application allows users to upload a mammogram image, select a pre-trained deep learning model
or ensemble mode, and receive a diagnostic prediction (benign or malignant) alongside Grad-CAM heatmaps
for visual explanation. Models are based on DenseNet, EfficientNet, and MobileNetV3 architectures, trained 
for binary classification. Grad-CAM visualisations highlight image regions contributing to model decisions.

Developed as part of a dissertation project on interpretable AI for mammogram classification.
"""

from flask import Flask, request, render_template

from PIL import Image
import numpy as np
import os

import matplotlib
# Use non-GUI backend
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torchvision.transforms.functional import to_pil_image

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import densenet169, densenet121, efficientnet_b0, mobilenet_v3_large, mobilenet_v3_small

# Flask app setup
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variables for Grad-CAM hooks
gradients = None
activations = None

def register_gradcam_hooks(last_conv_layer):
    """
    Register forward and backward hooks for Grad-CAM extraction
    from the last convolutional layer of the model.
    """
    global gradients, activations

    def forward_hook(module, args, output):
        global activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output

    last_conv_layer.register_forward_hook(forward_hook)
    last_conv_layer.register_full_backward_hook(backward_hook)

# Model loading functions

def load_model_d169():
    """
    Load DenseNet-169 model trained for binary classification 
    and register Grad-CAM hooks.
    """
    model_path = "static/models/best_model_run25.pth"
    model = densenet169(weights=None)
    model.classifier = nn.Sequential(
        #nn.Dropout(0.5),
        nn.Linear(in_features=1664, out_features=1)
    )
    # Load model state
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Register Grad-CAM hooks
    last_conv_layer = model.features.denseblock4.denselayer32.conv2
    register_gradcam_hooks(last_conv_layer)

    return model

def load_model_d121():
    """
    Load DenseNet-121 model trained for binary classification 
    and register Grad-CAM hooks.
    """
    model_path = "static/models/best_model_run13.pth"
    model = densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=1024, out_features=1)
    )
    # Load model state
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Register Grad-CAM hooks
    last_conv_layer = model.features.denseblock4.denselayer16.conv2
    register_gradcam_hooks(last_conv_layer)

    return model

def load_model_eb0():
    """
    Load EfficientNet-B0 model trained for binary classification 
    and register Grad-CAM hooks.
    """
    model_path = "static/models/best_model_run20.pth"
    model = efficientnet_b0(weights=None)
    # Modify the classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=1280, out_features=1)
    )
    # Load model state
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Register Grad-CAM hooks
    last_conv_layer = model.features[-1][0]
    register_gradcam_hooks(last_conv_layer)

    return model

def load_model_mnl():
    """
    Load MobileNetV3-Large model trained for binary classification 
    and register Grad-CAM hooks.
    """
    model_path = "static/models/best_model_run39.pth"
    model = mobilenet_v3_large(weights=None)
    
    # Check the number of input features for the classifier
    num_features = model.classifier[0].in_features
    # Modify the classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1)
    )

    # Load model state
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))

    # Move model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Register Grad-CAM hooks on the last convolutional layer
    last_conv_layer = model.features[-1][0]
    register_gradcam_hooks(last_conv_layer)

    return model

def load_model_mns():
    """
    Load MobileNetV3-Small model trained for binary classification 
    and register Grad-CAM hooks.
    """
    model_path = "static/models/best_model_run15.pth"
    model = mobilenet_v3_small(weights=None)
    
    # Check the number of input features for the classifier
    num_features = model.classifier[0].in_features
    # Modify the classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 1)
    )

    # Load model state
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))

    # Move model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Register Grad-CAM hooks on the last convolutional layer
    last_conv_layer = model.features[-1][0]
    register_gradcam_hooks(last_conv_layer)

    return model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to unnormalise images
def unnormalise(img_tensor, mean, std):
    """
    Revert normalisation applied to an image tensor.

    Args:
        img_tensor (torch.Tensor): Normalised tensor.
        mean (list): Channel-wise means used during normalisation.
        std (list): Channel-wise standard deviations used during normalisation.

    Returns:
        torch.Tensor: Un-normalised image tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean  # Reverse normalization

def generate_gradcam(model, img_tensor):
    """
    Generate a Grad-CAM heatmap for a given input tensor and model.

    Args:
        model: Trained CNN model.
        img_tensor (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Grad-CAM heatmap.
    """
    global gradients, activations

    model.zero_grad()
    output = model(img_tensor.unsqueeze(0))
    output.backward()

    # Pool gradients across channels
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

    # Weight activations by gradients
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Compute heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    # Normalise
    heatmap /= torch.max(heatmap)  

    return heatmap.detach().cpu()


# Flask web route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    """
    Handle image uploads, model selection, prediction, and Grad-CAM generation.
    """
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return "No file uploaded!", 400
        
        file = request.files["file"]

        # Check file extension
        if not file.filename.lower().endswith(".png"):
            return "Only PNG images are supported!", 400

        # Determine which model was selected (default to ensemble if no model is selected)
        selected_model = request.form.get("model", "ensemble")

        # Save uploaded image
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load and preprocess the image
        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img)

        # Unnormalise, resize and convert image
        original_size = img.size
        unnorm_img = unnormalise(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Resize the unnormalised image to the original size
        original_img = to_pil_image(unnorm_img.clamp(0, 1), mode='RGB').resize(original_size, resample=Image.BICUBIC)

        # Load the correct model
        if selected_model == "ensemble":
            models, model_names = [], []
            model_loaders = [
                ("DenseNet-169", load_model_d169),
                ("DenseNet-121", load_model_d121),
                ("EfficientNet-B0", load_model_eb0),
                ("MobileNetV3-Large", load_model_mnl),
                ("MobileNetV3-Small", load_model_mns),
            ]
            for name, loader in model_loaders:
                model_names.append(name)
                models.append(loader())
        else:
            model_dict = {
                "d169": load_model_d169,
                "d121": load_model_d121,
                "eb0": load_model_eb0,
                "mnl": load_model_mnl,
                "mns": load_model_mns
            }
            if selected_model not in model_dict:
                return "Invalid model selected!", 400
            model = model_dict[selected_model]()

        # Prediction and Grad-CAM visualisation
        if selected_model == "ensemble":
            predictions, confidences, ensemble_results = [], [], []

            for i, model in enumerate(models):
                model_name = model_names[i]
                
                with torch.no_grad():
                    pred = model(img_tensor.unsqueeze(0))
                    prob = pred.sigmoid()
                    pred_label = (prob > 0.5).float().item()
                    confidence = (prob.item() - 0.5) * 200 if pred_label == 1 else (0.5 - prob.item()) * 200
                    predictions.append(pred_label)
                    confidences.append(confidence)

                result_str = "Malignant" if pred_label == 1 else "Benign"
                confidence = round(confidence, 2)

                # Generate heatmap
                heatmap = generate_gradcam(model, img_tensor)
                
                # Resize heatmap and apply colormap
                overlay = to_pil_image(heatmap, mode='F').resize(original_size, resample=Image.BICUBIC)
                cmap = colormaps['jet']
                overlay_img = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
                
                # Overlay heatmap on the original image
                fig, ax = plt.subplots()
                ax.imshow(original_img)
                ax.imshow(overlay_img, alpha=0.4, interpolation='nearest')
                ax.axis('off')

                # Save each heatmap
                heatmap_filename = f"heatmap_{model_names[i].replace(' ', '_')}_{file.filename}"
                heatmap_path = os.path.join(RESULTS_FOLDER, heatmap_filename)
                plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                # Append all relevant info
                ensemble_results.append({
                    "model": model_name,
                    "prediction": result_str,
                    "confidence": confidence,
                    "heatmap": "static/results/" + heatmap_filename
                })

            # Get the majority class (1 if 3+ predicted positive)
            majority = int(sum(predictions) >= 3)
            # Filter confidences from models that agree with the majority
            majority_confidences = [
                conf for pred, conf in zip(predictions, confidences) if pred == majority
            ]
            # Compute average confidence from agreeing models only
            avg_confidence = round(sum(majority_confidences) / len(majority_confidences), 2)
            
            result = "Malignant" if majority == 1 else "Benign"

            return render_template("index.html", filename=file.filename, result=result, confidence=avg_confidence, model="ensemble", ensemble_results=ensemble_results)
        
        else:
            # Run the model
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0))
                prob = pred.sigmoid()
                pred_label = (prob > 0.5).float()
                if pred_label == 1:
                    result = "Malignant"
                    confidence = float((prob - 0.5) * 200)
                else:
                    result = "Benign"
                    confidence = float((0.5 - prob) * 200)

            confidence=round(confidence, 2)

            # Generate Grad-CAM heatmap
            heatmap = generate_gradcam(model, img_tensor)

            # Resize heatmap and apply colormap
            overlay = to_pil_image(heatmap, mode='F').resize(original_size, resample=Image.BICUBIC)
            cmap = colormaps['jet']
            overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

            # Overlay heatmap on the original image
            fig, ax = plt.subplots()
            ax.imshow(original_img)
            ax.imshow(overlay, alpha=0.4, interpolation='nearest')
            ax.axis('off')

            # Save Grad-CAM result
            heatmap_path = os.path.join(RESULTS_FOLDER, "heatmap_" + file.filename)
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            return render_template("index.html", filename=file.filename, result=result, confidence=confidence, heatmap=heatmap_path, model=selected_model)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
