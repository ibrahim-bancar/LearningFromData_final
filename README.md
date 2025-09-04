# Learning From Data – SVM & MLP (Final Project)

## Overview
- From-scratch implementation of **Linear SVM (SMO-based)** and **Multi-Layer Perceptron (MLP)**.  
- Environment: `Python 3.10+`  
- Experiments/Demos: `main.ipynb`  
- Visuals: located in the `figures/` folder.  


---

## 🚀 Quick Start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Open the notebook
jupyter notebook main.ipynb
```

> You can also directly import the Python modules instead of using the notebook. See usage examples below.

---

## 📦 Project Structure

```
lfd-final/
├─ main.ipynb
├─ requirements.txt
├─ src/
│  ├─ SVM.py
│  └─ MultiLayerPerceptron.py
└─ figures/
   ├─ datasets_visualization.png
   ├─ hard_margin_svm_d1.png
   ├─ mlp_d1.png
   ├─ mlp_d2.png
   └─ soft_margin_svm_d2.png
```

---

## 🧠 Modules and Usage

### 1) Linear SVM (SMO) – `src/SVM.py`

**Features:**
- Linear kernel  
- Hyperparameters: `C`, `tol`, `max_passes`, `max_iter`  
- API: `fit(X, y)` for training, `predict(X)` for inference  

**Example usage:**
```python
import numpy as np
from sklearn.datasets import make_blobs
from src.SVM import SVM

# Data
X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)
y = np.where(y == 0, -1, 1)  # SVM expects -1/+1 labels

# Model
svm = SVM(C=1.0, tol=1e-3, max_passes=5, max_iter=1000, random_state=42)
svm.fit(X, y)

# Accuracy
pred = svm.predict(X)
acc = (pred == y).mean()
print(f"Train Accuracy: {acc:.3f}")
```

---

### 2) Multi-Layer Perceptron – `src/MultiLayerPerceptron.py`

**Features:**
- Flexible layer definition via `hidden_sizes`  
- Supports `relu`, `tanh`, `sigmoid` activations (default output: `softmax`)  
- Parameters: `learning_rate`, `random_state`  
- API: `fit(X, y, epochs=...)`, `predict(X)`  

**Example usage:**
```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from src.MultiLayerPerceptron import MultiLayerPerceptron

# Data
X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
y = y.reshape(-1, 1)
enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y)

# Model
mlp = MultiLayerPerceptron(
    input_size=X.shape[1],
    hidden_sizes=[16, 16],
    output_size=Y.shape[1],
    activation='relu',
    learning_rate=0.01,
    random_state=0
)

mlp.fit(X, Y, epochs=200)

pred = mlp.predict(X)
acc = (pred.reshape(-1,1) == y).mean()
print(f"Train Accuracy: {acc:.3f}")
```

---

## 📊 Results and Visuals

The `figures/` folder contains example outputs:
- `datasets_visualization.png`: dataset visualization  
- `hard_margin_svm_d1.png` & `soft_margin_svm_d2.png`: SVM margin examples  
- `mlp_d1.png`, `mlp_d2.png`: MLP decision boundaries  

> You can embed these visuals in the notebook or generate new ones with `matplotlib`.

---
 


---

## 📄 License

Add a license file (e.g., `LICENSE`). MIT is recommended for educational/research use.

---

## ✉️ Contact

- Name: İbrahim BANCAR
- Email: bancar22@itu.edu.tr
