# Fake News Detection using Machine Learning

A machine learning project that classifies news articles as **Fake** or **Credible** using a dataset related to the Syrian War. The project applies the machine learning workflow which includes data preparation and feature creation to model training and prediction.

## 📌 Project Purpose

The purpose of this project is to demonstrate how machine learning can be applied to a real-world classification problem: **identifying whether a news item is fake or credible**.

The project uses a Syrian-war news dataset obtained from **Kaggle**, with credibility labels based on **VDC ground-truth information**.

The project demonstrates the following machine learning workflow:

**Data Preparation → Feature Creation → Model Training → Prediction**

## ✨ Key Features

* Classification of news into two categories:

  * `0` — Fake
  * `1` — Credible
* Data preparation and cleaning
* Feature creation and preparation
* Machine learning model training
* Prediction of news credibility
* Evaluation of classification performance
* Interactive web-based interface using Streamlit
* Deployed as a web application

## 🛠️ Technology Stack

### Programming Language

* **Python**

### Libraries & Frameworks

* **Pandas** — Data handling and manipulation
* **Scikit-learn** — Machine learning, model training and evaluation
* **Streamlit** — Web application and deployment

### Data Source

* **Kaggle** — Syrian-war news dataset
* **VDC** — Ground-truth information used for credibility labels

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
│__ index.html
```

> The exact file and folder names may vary depending on the final project structure.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd Fake-News-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Verify the Dataset and Model Files

Make sure the required dataset and trained model files are present in their respective directories before running the application.

## 🚀 Usage

### Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL in the terminal.

Open the URL in your browser to access the application.

### Using the Application

1. Open the Streamlit application.
2. Provide the required news input.
3. Submit the input for prediction.
4. The trained machine learning model processes the input.
5. The application displays the predicted credibility class:

   * **0 — Fake**
   * **1 — Credible**

## 🔄 Machine Learning Workflow

The project follows a structured machine learning development cycle:

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Cleaning & Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction
        ↓
Deployment
```

This workflow demonstrates how the concepts studied during the machine learning course can be combined to build a practical application.

## 📊 Model Evaluation

The project uses standard classification evaluation measures to assess model performance, including:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Validation and model tuning concepts covered in the project include:

* Cross-Validation
* Grid Search CV
* Randomized Search CV

## 🌐 Deployment

The trained machine learning model has been deployed as a **Streamlit web application**(https://fake-news-detection-model-byaditya.streamlit.app/?utm_source=chatgpt.com), allowing users to interact with the project through a browser rather than running the complete machine learning workflow manually.

## 🔮 Future Improvements

The project can be further improved by:

* Improving the overall prediction performance
* Experimenting with additional machine learning algorithms
* Improving feature selection and preparation
* Enhancing the Streamlit user interface
* Adding better visualization of model predictions and evaluation results
* Providing more detailed prediction results to users
* Expanding the dataset with additional verified news sources
* Improving the deployment and scalability of the application

## 🎓 Learning Outcomes

Through this project, the following machine learning concepts were applied:

* Data Collection
* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* Feature Engineering & Selection
* Supervised Learning
* Classification
* Model Training
* Model Evaluation
* Model Validation
* Hyperparameter Tuning
* Model Deployment

## 👨‍💻 Author

**Aditya Kumar**

Python • Machine Learning • Data Analysis
