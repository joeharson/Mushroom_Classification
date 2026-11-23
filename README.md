# 🍄 Mushroom Edibility & Toxicity AI Classifier

An end-to-end Machine Learning web application and mycology diagnostic suite built with **Streamlit** and **Scikit-Learn** (Random Forest Classifier, 99.1%+ empirical test accuracy) to predict whether a wild mushroom specimen is **Edible** or **Poisonous** based on 22 morphological and environmental characteristics.

---

## 🚀 Execution Flow

Follow this 2-step workflow to train the model and launch the web interface:

```
┌─────────────────────────┐       Generates       ┌───────────────────────────┐
│   Step 1: Model Training│ ───────────────────>  │  model.pkl                │
│   Model_pkl.ipynb       │                       │  label_encoder.pkl        │
└─────────────────────────┘                       └───────────────────────────┘
                                                                │
                                                                ▼
┌─────────────────────────┐       Launches        ┌───────────────────────────┐
│   Step 2: Streamlit UI  │ <───────────────────  │  http://localhost:8501    │
│   streamlit run app.py  │                       │  (Overview & pages/app1)  │
└─────────────────────────┘                       └───────────────────────────┘
```

### Step 1: Train the Machine Learning Model

Run the Jupyter Notebook [`Model_pkl.ipynb`](Model_pkl.ipynb) to train and export the classifier:

1. Open `Model_pkl.ipynb` in VS Code or JupyterLab:
   ```bash
   jupyter notebook Model_pkl.ipynb
   ```
2. Execute the notebook cells sequentially:
   * **Data Ingestion**: Loads the UCI mushroom dataset (`mushrooms.csv`, 8,124 samples).
   * **Preprocessing & Encoding**: Encodes categorical attributes.
   * **Model Training**: Fits a `RandomForestClassifier` with 100 decision trees.
   * **Evaluation**: Generates accuracy score, confusion matrix, and classification report.
   * **Serialization**: Exports the trained models as `model.pkl` and `label_encoder.pkl`.

### Step 2: Launch the Streamlit Web Application

Once `model.pkl` and `label_encoder.pkl` are generated, start the interactive web application using `app.py`:

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

* **Main Page (`app.py`)**: Project background, dataset distribution metrics, ML pipeline architecture, and dataset table preview.
* **Classifier Page (`pages/app1.py`)**: Multi-page diagnostic tool with:
  * Categorized tabs (Cap & Odor, Gills & Spores, Stalk & Ring, Habitat & Ecology).
  * Human-readable plain English dropdown selections (words alone, no cryptic single-character codes).
  * Quick-load presets (**Safe Edible**, **Highly Toxic**, and **Reset Defaults**).
  * Real-time edibility probability vs. toxicity hazard assessment and risk breakdown.

---

## 🐳 Docker Deployment (Zero-Install Method)

Run the entire application in an isolated container without needing a local Python setup:

### Using Docker Compose (Recommended)

```bash
docker compose up --build
```

Access the application at `http://localhost:8501`.

To stop the container:
```bash
docker compose down
```

### Using Standard Docker CLI

```bash
# Build the Docker image
docker build -t mushroom-ai-app .

# Run the container
docker run -p 8501:8501 mushroom-ai-app
```

---

## 📁 Project Structure

```text
├── app.py                     # Main application entrypoint (Home & Overview)
├── pages/
│   └── app1.py                # Specimen Classifier (Interactive UI)
├── Model_pkl.ipynb            # Jupyter Notebook to train and save the model
├── model.pkl                  # Trained Random Forest ensemble classifier
├── label_encoder.pkl          # Target class encoder
├── mushrooms.csv              # UCI Mushroom benchmark dataset (8,124 rows, 22 features)
├── requirements.txt           # Python package dependencies
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Container service configuration
├── .dockerignore              # Docker build exclusions
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation and guide
```

---

## 🛠️ Local Installation & Development Setup

If running locally with virtual environments:

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if retrain needed)
jupyter execute Model_pkl.ipynb

# 4. Launch the web application
streamlit run app.py
```

---

## ⚠️ Disclaimer

This application is designed for educational and research purposes. Never consume wild-foraged mushrooms based solely on automated classifiers. Many edible species have deadly toxic lookalikes in nature. Always consult certified mycological experts before consuming wild fungi.
