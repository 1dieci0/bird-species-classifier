# Bird Species Classifier 🐦

A deep learning image classification project that identifies bird species from images using **transfer learning with PyTorch**.

The model is trained on the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset, containing 200 bird species, and uses a pretrained **ResNet-18** architecture fine-tuned for fine-grained bird classification.

---

## Project Overview

The goal of this project is to build an end-to-end computer vision pipeline capable of classifying bird images into one of 200 species.

The project includes:

- Custom PyTorch dataset implementation
- Dataset preprocessing and metadata handling
- Data augmentation
- Transfer learning with a pretrained CNN
- Training and validation pipeline
- Model checkpointing
- Test evaluation
- Single-image inference

---

## Results

Final evaluation on the test set:

| Metric | Result |
|---|---:|
| Test Accuracy | **72.25%** |
| Number of Classes | 200 |
| Model | ResNet-18 |

The model is able to recognize fine-grained visual differences between bird species, including:

- Color patterns
- Body shape
- Feather characteristics
- Beak structure
- Species-specific features

---

## Dataset

This project uses:

**Caltech-UCSD Birds-200-2011 (CUB-200-2011)**

Dataset characteristics:

- 200 bird species
- 11,788 images
- Bounding box annotations
- Part locations
- Attribute annotations
- Official train/test split

Dataset website:

https://www.vision.caltech.edu/datasets/cub_200_2011/

The dataset is not included in this repository due to size and licensing restrictions.

After downloading, place it in:

```
data/
└── CUB_200_2011/
```

---

## Model Architecture

The classifier uses:

**ResNet-18 pretrained on ImageNet**

The original ImageNet classifier was replaced:

```
Original:
ResNet-18
    ↓
Linear(512, 1000)

Modified:
ResNet-18
    ↓
Linear(512, 200)
```

The convolutional layers provide pretrained visual features, while the final layer is fine-tuned to classify the 200 bird species.

---

## Training Pipeline

### Data preprocessing

Training images use augmentation:

- Random resized crops
- Horizontal flips
- Random rotations
- Color jitter

Validation and test images use deterministic preprocessing.

---

### Optimization

The model was trained using:

- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate scheduling
- Model checkpointing based on validation accuracy

---

## Repository Structure

```
bird-species-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   └── best_bird_model.pth
│
├── notebooks/
│   └── bird_classifier.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── results/
    ├── training_curve.png
    └── sample_predictions.png
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>

cd bird-species-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

To train the model:

```bash
python src/train.py
```

The best-performing model will be saved as:

```
models/best_bird_model.pth
```

---

## Inference

To classify a new bird image:

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```
Prediction:
001.Black_footed_Albatross

Confidence:
92.4%
```

---

## Example Predictions

(Add prediction images here)

Example:

```
Input image
     ↓
Model
     ↓
Prediction + confidence score
```

---

## Future Improvements

Possible improvements:

- Experiment with stronger architectures:
  - ResNet-50
  - EfficientNet
  - Vision Transformers

- Add top-k predictions
- Add an interactive web demo using Gradio or Streamlit
- Improve out-of-distribution detection for bird species not present in CUB-200-2011
- Compare different augmentation strategies

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## License

This project uses the CUB-200-2011 dataset, which has its own license and usage terms.

Code in this repository is provided for educational and research purposes.
