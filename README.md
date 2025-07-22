# 💼 Employee Salary Prediction Web App

This project is a machine learning-powered web application that predicts whether an employee earns more than $50K per year, based on demographic and work-related inputs. It is built using Python, Streamlit, and trained on the UCI Adult dataset.

---

## 🚀 Features

- Predicts employee salary class: `>50K` or `<=50K`
- Interactive user interface using Streamlit
- Real-time predictions from single inputs
- Option to deploy and run in Google Colab with `pyngrok`
- Model trained and saved using scikit-learn

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Description:** Census data including features like age, workclass, education, occupation, hours-per-week, etc.
- **Target:** `income` (`<=50K` or `>50K`)

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Framework:** Streamlit
- **Modeling:** scikit-learn
- **Visualization:** pandas, matplotlib
- **Deployment (Optional):** pyngrok + Google Colab

---

## ⚙️ How to Run

### 🔧 Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/employee-salary-predictor.git
   cd employee-salary-predictor
2. pip install -r requirements.txt

3. streamlit run app.py
