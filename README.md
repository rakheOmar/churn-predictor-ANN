# Customer Churn Prediction 🤖

A simple Streamlit web app that predicts customer churn using an Artificial Neural Network (ANN).

> Made this while learning Deep Learning from [Krish Naik](https://www.youtube.com/@krishnaik06)

## 📸 Screenshot

<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/7b3020d8-ce5f-4c30-8543-b3aeef8cfff9" />

## 🛠️ Tech Stack

- **Python 3.13**
- **TensorFlow** - Deep learning model
- **Streamlit** - Web interface
- **scikit-learn** - Data preprocessing
- **uv** - Fast Python package manager
- **ruff** - Fast Python linter

## 📦 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/rakheOmar/churn-predictor-ANN.git
   cd churn-predictor-ANN
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

## 🚀 Run the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

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

## 🔧 Development

**Lint code with ruff:**

```bash
ruff check .
```

**Format code with ruff:**

```bash
ruff format .
```

## 💡 Usage

1. Enter customer information in the left panel (split into 2 columns)
2. Click **"Predict Churn"** button
3. View the prediction results on the right panel

## 🧠 Model Details

The model is trained on the Churn Modelling dataset and uses:

**Input Features:**

- Geography (France, Spain, Germany)
- Gender (Male, Female)
- Age
- Credit Score
- Account Balance
- Tenure (years with bank)
- Number of Products
- Has Credit Card (Yes/No)
- Is Active Member (Yes/No)
- Estimated Salary

**Output:**

- Churn probability (0-1)
- Risk level (Low/Medium/High)


## 🙏 Acknowledgments

- Dataset: [Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- Tutorial by: [Krish Naik](https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/)

---

Made with ❤️ while learning Deep Learning
