# KongPyeng

> From “공평 (fairness)” in Korean, **KongPyeng** reflects our mission to develop AI that is accurate, unbiased, and ethically grounded in decision-making.

## 🛠️ How to Run Our Model

This project performs multi-view product recognition using YOLO and evaluates results under various parameter settings.

---

### 1. Run Raw Cache Generator

```
python cache_raw.py
```

- Runs YOLO inference on the dataset and generates `.txt` files containing labels and bounding box center coordinates `(cx, cy)`.
- Output files are saved to:  
  `/home/aistore51/git/output/`

---

### 2. Run Grid Search Over Parameters

```
python main_gridsearch.py
```

- Loads the raw `.txt` results and evaluates multiple parameter combinations:
  - Confidence thresholds
  - Minimum number of matching views
  - Aggregation strategies

---

### 3. Run Final Evaluation

```
python evaluating.py
```

- Compares grid search results with ground truth to compute **Mean Absolute Errors (MAE)**.
- Lower MAE indicates better performance.
- Evaluation results are saved to:  
  `/home/aistore51/git/eval_results.csv`

---

## 📂 Directory Structure

```
/home/aistore51/git/
├── cache_raw.py          # Caches YOLO inference results
├── main_gridsearch.py    # Parameter sweep and result generation
├── evaluating.py         # MAE evaluation against ground truth
├── output/               # Folder for cached raw results
└── eval_results.csv      # Final evaluation summary
```

---

## 📋 Requirements

- Python 3.8 ~ 3.10
- [SuperGradients](https://github.com/Deci-AI/super-gradients) library (for YOLO-NAS)
- torch >= 1.12.0
- torchvision
- opencv-python
- numpy
- tqdm
- PyYAML
- onnx, onnxruntime
- matplotlib
- albumentations
- pandas