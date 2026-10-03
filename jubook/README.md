# Jubook: Kaggriculture Online Cloud Training & Encrypted Submission

This directory contains the self-contained Jupyter Notebook for researching, online background training on Kaggle's free compute, encrypting your code, and submitting to the competition.

### Files
- **`kaggriculture_hypertrain_pipeline.ipynb`**: The primary standalone notebook.
- **`build_notebook.py`**: Automated script that re-compiles the notebook whenever you update `main.py`.

---

### How to Train Online on Kaggle (Background Execution)

Kaggle allows notebooks to run **in the background for up to 12 hours** on their cloud servers with **4 dedicated CPU cores**. You do NOT need to keep your browser or computer open.

1. **Upload Notebook**:
   - Go to **[kaggle.com/code](https://www.kaggle.com/code)** -> Click **"New Notebook"**.
   - Select **File** -> **Upload Notebook** and choose `kaggriculture_hypertrain_pipeline.ipynb`.
2. **Configure Duration**:
   - In Section 4 (Cell 8), set `TRAIN_HOURS = 2.0` (or `4.0` or `8.0` hours).
3. **Start Background Run (Commit)**:
   - In the top right corner, click **"Save Version"**.
   - Select **"Save & Run All (Commit)"** and click **Save**.
4. **Walk Away**:
   - You can close your laptop. Kaggle will run the evolutionary training across rotating seeds, test candidate policies, and save `best_config.json`.
   - On every new peak score, it automatically re-encrypts the agent and rebuilds `submission.tar.gz`.
5. **Submit Best Agent**:
   - When finished (or anytime you view the saved notebook version), open the **Output** panel on the right sidebar.
   - Click **Submit to Competition** on `submission.tar.gz`.

---

### How to Run Locally
```bash
jupyter lab jubook/kaggriculture_hypertrain_pipeline.ipynb
# or
jupyter notebook jubook/kaggriculture_hypertrain_pipeline.ipynb
```
