# 📊 Feature Engineering Impact Analyzer

A **Streamlit** application that demonstrates the impact of **feature engineering** on machine learning performance by comparing a baseline model trained on raw data with the same model trained on an engineered dataset.

The application provides an interactive workflow for data preprocessing, feature engineering, model training, and performance evaluation, allowing users to observe how preprocessing techniques affect model performance.

---

## 🎥 Demo

<p align="center">
  <img src="assets/demo.gif" width="100%">
</p>

## 🌐 Live Demo

🔗 https://feature-engineering-analyzer-slr4etpc88hhfg6b3ouahx.streamlit.app/

---

## ✨ Features

### 📂 Dataset Management
- Upload any CSV dataset
- Automatic target column validation
- Interactive data preview
- Dataset statistics and summary

### 🛠 Data Preprocessing
- Missing value handling
- Outlier detection and treatment
- Feature scaling
  - Standard Scaling
  - Min-Max Scaling
  - Robust Scaling
- Categorical feature encoding
  - Label Encoding
  - One-Hot Encoding

### 📊 Feature Engineering
- Correlation heatmap visualization
- Feature selection
- Engineered dataset generation

### 🤖 Machine Learning
- Train multiple machine learning algorithms
- Baseline model using raw data
- Model training using engineered data
- Side-by-side model comparison

### 📈 Evaluation & Export
- Model evaluation metrics
- Performance comparison
- Download engineered dataset
- Download trained model

---

## ⚙️ Workflow

```text
                 Upload CSV Dataset
                         │
                         ▼
              Select Target Column
                         │
                         ▼
             Dataset Preview & Statistics
                         │
                         ▼
                Data Preprocessing
     ┌─────────────────────────────────────┐
     │ • Missing Value Handling            │
     │ • Outlier Detection                 │
     │ • Feature Scaling                   │
     │ • Categorical Encoding              │
     └─────────────────────────────────────┘
                         │
                         ▼
               Feature Engineering
                         │
                         ▼
             Baseline Model Training
               (Original Dataset)
                         │
                         ▼
            Engineered Model Training
              (Processed Dataset)
                         │
                         ▼
            Performance Comparison
                         │
                         ▼
       Download Model & Engineered Dataset
```

---

## 📸 Screenshots

### Data Preview

<img src="assets/preview.png" width="100%">

### Final Results

<img src="assets/results.png" width="100%">

---

## 📁 Project Structure

```text
feature-engineering-analyzer/
│
├── assets/
│   ├── demo.gif
│   ├── preview.png
│   └── results.png
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Framework | Streamlit |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sushantkumar31/feature-engineering-analyzer.git
```

### 2️⃣ Navigate to the Project Directory

```bash
cd feature-engineering-analyzer
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📊 Sample Dataset

You can test the application using the Titanic dataset.

**Dataset**

https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

**Target Column**

```text
Survived
```

---

## 💡 Why This Project?

Feature engineering is one of the most important steps in the machine learning pipeline, but its contribution to model performance is often difficult to quantify.

This application demonstrates its impact by training the **same machine learning algorithm** under two different conditions:

- **Baseline Model** – trained using the original dataset.
- **Engineered Model** – trained after applying preprocessing and feature engineering techniques.

Keeping the learning algorithm unchanged allows users to focus on how feature engineering influences evaluation metrics and overall model performance.

---

## 🎯 Use Cases

- Learning machine learning preprocessing techniques
- Understanding feature engineering workflows
- Comparing model performance before and after preprocessing
- Educational demonstrations
- Portfolio and resume projects
- Rapid experimentation with tabular datasets

---

## 📦 Dependencies

- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome!

If you would like to improve the project:

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is available under the MIT License.

If you haven't added a license yet, you can either add an MIT License or remove this section.

---

## ⭐ Support

If you found this project useful, please consider giving it a **⭐ on GitHub**.

Your support helps the project reach more developers and motivates future improvements.

Happy Learning! 🚀
