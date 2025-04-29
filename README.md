# dissertation-mammogram-classification
Dissertation Project: An interpretable deep learning pipeline for binary classification of mammograms, using CNNs, Grad-CAM visualisations, and transfer learning. Includes a local web interface for model testing and explanation.

## Running the code and webapp

### Setting up the virtual environment

```shell
# Install virtualenv package
pip install virtualenv

# In the directory containing the source code,
# create the virtual environment
python3.11 -m venv env

# Activate virtual environment and install requirements.txt

# For MacOS/Linux:
source env/bin/activate

# You should see (env) before the command prompt
pip install -r requirements.txt
```

### Starting the webapp

Activat the virtual environment:

```shell
source env/bin/activate
```

Move in the [webapp](./mammogram-ai-project/webapp/), and start the webapp as shown:

```shell
cd mammogram-ai-project/webapp
python3.11 app.py
```