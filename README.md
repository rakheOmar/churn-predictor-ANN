# Customer Churn Prediction

A simple Streamlit web app that predicts customer churn using an Artificial Neural Network (ANN).

## Features

- Predicts customer churn probability
- Interactive web interface
- Real-time predictions using TensorFlow

## Tech Stack

- **Python 3.13**
- **TensorFlow** - Deep learning model
- **Streamlit** - Web interface
- **scikit-learn** - Data preprocessing
- **uv** - Fast Python package manager
- **ruff** - Fast Python linter

## Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd "Section 53 - End to End Deep Learning Project Using ANN"
   ```

2. **Install dependencies using uv**

   ```bash
   uv sync
   ```

3. **Activate the virtual environment**

   ```bash
   # On Windows PowerShell
   .\ann-project\Scripts\activate.ps1

   # On Unix/MacOS
   source ann-project/bin/activate
   ```

## Run the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
├── main.py                    # Streamlit application
├── models/                    # Trained model and preprocessors
│   ├── model.keras
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   └── scaler.pkl
├── data/                      # Dataset
│   └── Churn_Modelling.csv
├── notebooks/                 # Jupyter notebooks
│   ├── experiments.ipynb
│   └── predictions.ipynb
└── pyproject.toml            # Project dependencies
```

## Development

**Lint code with ruff:**

```bash
ruff check .
```

**Format code with ruff:**

```bash
ruff format .
```

## Usage

1. Enter customer information in the left panel
2. Click "Predict Churn" button
3. View the prediction results on the right panel

## Model

The model is trained on the Churn Modelling dataset and uses:

- Input features: Geography, Gender, Age, Credit Score, Balance, Tenure, Number of Products, Credit Card status, Active Member status, Estimated Salary
- Output: Churn probability (0-1)

## License

MIT
