# Stroke Risk Prediction - Deployment Guide

## Files You Need to Upload to GitHub

Upload these **5 files** to your GitHub repository:

### 1. Application Files
| File | Description |
|------|-------------|
| `streamlit_app.py` | Main Streamlit application (the UI) |
| `requirements.txt` | List of Python packages needed |

### 2. Model Files (from your Colab notebook)
| File | Description |
|------|-------------|
| `stroke_model.pkl` | Trained Logistic Regression model |
| `std_scaler.pkl` | StandardScaler for age normalization |
| `minmax_scaler.pkl` | MinMaxScaler for glucose/BMI |
| `features.pkl` | Feature column names in correct order |

---

## Step-by-Step Deployment Guide

### Step 1: Get Your Model Files from Colab

1. Open your `STROKE_PROJECT_(2).ipynb` notebook in Google Colab
2. Run all cells until the end (where model files are saved)
3. Download these files from Colab:
   - `stroke_model.pkl`
   - `std_scaler.pkl`
   - `minmax_scaler.pkl`
   - `features.pkl`

**How to download from Colab:**
- Click the folder icon on the left sidebar
- Right-click on each `.pkl` file
- Select "Download"

---

### Step 2: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click the **+** icon → **New repository**
3. Name it: `stroke-prediction-app`
4. Make it **Public** (required for free Streamlit Cloud)
5. Click **Create repository**

---

### Step 3: Upload Files to GitHub

**Option A: Using GitHub Website (Easiest)**

1. Go to your repository page
2. Click **" Add file" → "Upload files"**
3. Drag and drop all 6 files:
   ```
   streamlit_app.py
   requirements.txt
   stroke_model.pkl
   std_scaler.pkl
   minmax_scaler.pkl
   features.pkl
   ```
4. Add commit message: "Initial commit"
5. Click **"Commit changes"**

**Option B: Using Git Command Line**

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/stroke-prediction-app.git
cd stroke-prediction-app

# Copy your files to this folder
cp /path/to/streamlit_app.py .
cp /path/to/requirements.txt .
cp /path/to/stroke_model.pkl .
cp /path/to/std_scaler.pkl .
cp /path/to/minmax_scaler.pkl .
cp /path/to/features.pkl .

# Upload to GitHub
git add .
git commit -m "Add stroke prediction app"
git push origin main
```

---

### Step 4: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"Sign in with GitHub"**
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: Select `stroke-prediction-app`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.11
5. Click **"Deploy!"**

---

### Step 5: Your App is Live!

After a few minutes, you'll get a public URL like:
```
https://stroke-prediction-app-username.streamlit.app
```

Share this URL with anyone - they can use your app!

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure `requirements.txt` includes all needed packages

### Issue: "FileNotFoundError: stroke_model.pkl"
**Solution**: Upload all `.pkl` files to your GitHub repository

### Issue: App won't start
**Solution**:
1. Check the logs in Streamlit Cloud
2. Ensure file names match exactly (case-sensitive)
3. Verify `streamlit_app.py` is in the root directory

### Issue: Model gives wrong predictions
**Solution**:
1. Re-download model files from Colab
2. Make sure you ran the full training notebook
3. Re-upload `.pkl` files to GitHub

---

## Project Structure

Your GitHub repository should look like this:
```
stroke-prediction-app/
├── streamlit_app.py          # Main app (provided)
├── requirements.txt          # Dependencies (provided)
├── stroke_model.pkl          # Model (from Colab)
├── std_scaler.pkl            # Scaler (from Colab)
├── minmax_scaler.pkl         # Scaler (from Colab)
└── features.pkl              # Features (from Colab)
```

---

## Quick Reference

### Model Performance
- Algorithm: Logistic Regression
- Accuracy: 79.2%
- Recall: 48.5%
- F1-Score: 14.9%

### Input Features Required
1. Age (years)
2. Gender (Male/Female/Other)
3. Hypertension (Yes/No)
4. Heart Disease (Yes/No)
5. Ever Married (Yes/No)
6. Work Type
7. Residence Type (Urban/Rural)
8. Average Glucose Level (mg/dL)
9. BMI
10. Smoking Status

---

## Need Help?

- Streamlit docs: https://docs.streamlit.io
- GitHub docs: https://docs.github.com
- Model training: Check your Colab notebook

Good luck with your deployment! 🚀
