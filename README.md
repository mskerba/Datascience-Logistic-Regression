# 🧙‍♂️ Datascience X Logistic Regression — *Harry Potter and a Data Scientist*

Recreate the legendary Sorting Hat using data science and machine learning magic!
This project implements a **multi-class logistic regression** model from scratch to classify Hogwarts students into their respective houses.

---

## 🎯 Project Overview

Professor McGonagall needs your help — the Sorting Hat has stopped working!
Your mission is to build a new one using muggle data science and machine learning.

### Core Steps

- Data analysis implemented from scratch (manual count/mean/std/min/max)
- Data visualization (histogram, scatter, pair plots)
- Implement one-vs-all logistic regression trained with gradient descent
- Predict houses for new students and export results to CSV

---

## 🧠 Learning Objectives

- Understand data preprocessing and visualization
- Implement logistic regression manually and optimize a loss function
- Learn gradient descent and model evaluation

---

## 🧩 Bonus Ideas

- Add more statistical metrics to your describe function
- Try stochastic or mini-batch gradient descent
- Implement different optimization algorithms

---

## 🧾 Output Example

```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Slytherin
```

Your goal: achieve high accuracy and a robust model for the Sorting Hat.

---

## ⚙️ Technologies

- Python (recommended)
- NumPy, Pandas, Matplotlib, Seaborn
- No high-level ML training libraries for the core algorithm

---

## ▶️ How to run

Run the following commands from the project root (where this `README.md` is).

### Prerequisites

- Python 3.8+ (use the `python3` command)

### Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train the model (creates `thetas.csv`)

```bash
# uses datasets/dataset_train.csv by default
python3 logreg_train.py
# or explicitly
python3 logreg_train.py datasets/dataset_train.csv
```

The script prints training accuracy for each one-vs-all classifier and writes `thetas.csv` in the repo root.

### Predict using the trained thetas (creates `houses.csv`)

```bash
python3 logic_prediction.py datasets/dataset_test.csv --thetas thetas.csv --out houses.csv
```

This produces a CSV with columns `Index,Hogwarts House`.

### Visualizations

```bash
python3 histogram.py
python3 scatter_plot.py
python3 pair_plot.py
```

### Troubleshooting

- If `thetas.csv` is not found, run the training step first.
- Run commands from the project root so relative paths like `datasets/...` resolve.
- If a script errors about missing columns, check CSVs in `datasets/` and ensure they are not empty.

---

## 🪄 License

This project is for educational purposes and inspired by the *Harry Potter and a Data Scientist* exercise.

If you used the separate HOW_TO_RUN.md, it remains in the repo as a copy of these instructions.
