# **2.5D Liver and Tumor Segmentation**

This project implements a 2-stage (Liver -> Tumor) cascade using a 2.5D U-Net for segmenting the liver and tumors from 3D CT scans.

The full pipeline includes data conversion from PNGs, data preprocessing, a 2-stage model training process, full evaluation, and a web application for inference.

ALL USED DATASET ARE PUBLICALLY AVAILABLE
LITS CHALLENGE DATASET: https://competitions.codalab.org/competitions/17094 (REQUIRES PARTICIPATION)
3D-IRCAD-b1 DATASET: https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/

## **Project Structure**
```
LITS_Final_Project/
│
├── 3Dircadb1/                 (NIfTI volumes, *output* of Step 0)
├── LITS_Challenge_Data/       (Original raw .nii data, *input* for Step 0)
│
├── data_2_5d_liver/         (Processed slices for liver model)
├── data_2_5d_tumor_GT/      (Processed slices for tumor model)
│
├── trained_models/          (Final .pth models are saved here)
├── results/                 (Final .csv and plots are saved here)
├── web_app/                 (Contains the Flask web demo)
│   ├── app.py
│   ├── inference_worker.py
│   └── index.html
│
├── 3Dircadb1_data_conversion.py (Step 0: Converts PNGs to NIfTI)
├── model_unet_2.5d.py           (U-Net architecture)
│
├── 01_preprocess_liver.py       (Step 1a: Create liver slices)
├── 01_preprocess_tumor.py       (Step 1b: Create tumor slices)
├── 02_train_liver.py            (Step 2a: Train liver model)
├── 02_train_tumor.py            (Step 2b: Train tumor model)
├── 03_evaluate_cascade.py       (Step 3: Run full evaluation)
├── 04_analyze_results.py        (Step 4: Analyze CSV)
├── 05_predict_single.py         (Step 5: Visualize a single scan)
└── README.md                    
```

## **Full Pipeline Workflow**

### **Step 0: Convert Original Data (Run Once)**

This script converts the original PNG data from `LITS_Challenge_Data` into NIfTI volumes and saves them in the `3Dircadb1` directory, which is what all other scripts expect.
```
# This script reads from LITS_Challenge_Data/ and writes to 3Dircadb1/
python 3Dircadb1_data_conversion.py
```

### **Step 1: Preprocess NIfTI Data (Run Once)**

These scripts read the NIfTI files from `3Dircadb1` and create the 2.5D `.npy` slices for training.
```
# 1a. Create 2.5D slices for the Liver model
# Reads from 3Dircadb1/ -> Saves to data_2_5d_liver/
python 01_preprocess_liver.py

# 1b. Create 2.5D slices for the Tumor model (tumor-only slices)
# Reads from 3Dircadb1/ -> Saves to data_2_5d_tumor_GT/
python 01_preprocess_tumor.py
```

### **Step 2: Train Models**

These scripts train the models using the preprocessed data and save the best models to the `trained_models/` folder.
```
# 2a. Train the Liver model
# Reads from data_2_5d_liver/ -> Saves to trained_models/
python 02_train_liver.py

# 2b. Train the Tumor model (with Focal Loss)
# Reads from data_2_5d_tumor_GT/ -> Saves to trained_models/
python 02_train_tumor.py
```

### **Step 3: Evaluate Models**

This script runs the full 2-stage cascade on the entire `3Dircadb1` dataset and saves the final metrics as a CSV file in the `results/` folder.
```
# Reads from 3Dircadb1/ and trained_models/ -> Saves to results/
python 03_evaluate_cascade.py
```

### **Step 4: Analyze Results**

This script reads the CSV file from `results/` and prints the detailed analysis tables (performance by size, hard cases, etc.) for your report.
```
# Reads from results/eval_results_2_stage_full_leaderboard.csv
python 04_analyze_results.py
```

### **Step 5: Visualize a Single Scan**

You can run this script at any time to generate 2D and 3D plots for a specific scan. Remember to edit the `SCAN_ID_TO_PREDICT` variable inside the script first.
```
# Reads from 3Dircadb1/ and trained_models/ -> Saves plots to results/
python 05_predict_single.py
```

## **How to Run the Web App**

The web app runs a Flask server to provide an interactive demo.
```
# 1. Navigate into the web_app directory
cd web_app

# 2. Run the Flask server
# (Ensure your Python environment with torch, flask, etc. is active)
python app.py

# 3. Open your browser and go to:
# http://localhost:5000
```

## **Recommendation**
Your can also run this using Docker.
```
# 1. Navigate into the web_app directory
cd web_app

# 2a. Install docker desktop and start the docker engine
# 2b. Run docker compose
docker compose build

# 3. Open your browser and go to:
# http://localhost:5000
```
