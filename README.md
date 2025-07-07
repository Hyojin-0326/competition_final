# KongPyeng

> From “공평 (fairness)” in Korean, **KongPyeng** reflects our mission to develop AI that is accurate, unbiased, and ethically grounded in decision-making.

## 🛠️ How to Run Our Model

This project performs multi-view product recognition using YOLO and evaluates results under various parameter settings.
first, cd to '/home/aistore51/git' directory.
for 

### 0. (Opional) Run Custom Postprocessing
```
python main.py --idea 1 --win 5 --k 1 --cached
python main.py --idea 1 --win 5 --k 1
```

- Applies a selected postprocessing strategy (`--idea`) followed by temporal smoothing (`--win`) and optional hysteresis filtering (`--k`).
- Uses cached YOLO results if available. Otherwise, re-runs inference.
- Outputs per-event class count results in the format:
saved as:  
`/home/aistore51/git/output/event_XXXXX.txt`

**Flags:**

- `--idea`: Postprocessing strategy (1 to 5)
- `--win`: Smoothing window size (int 3, 5, 7)
- `--k`: Hysteresis threshold `K` (0, 1, 2)
- `--cached`: Optional flag to use existing cache only (skip re-inference)

**Examples:**

Use existing cached detection results: `python main.py --idea 1 --win 5 --k 1 --cached`

Force re-inference and overwrite cache: `python main.py --idea 1 --win 5 --k 1`


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
├── outputs/              # gridsearched results
├── results/              # Folder for cached raw results
├── main.py               # Custom postprocessing & smoothing pipeline
├── custom_output/        # Custom postprocessed results
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