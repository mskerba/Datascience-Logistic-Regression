# 🧙‍♂️ Datascience X Logistic Regression — *Harry Potter and a Data Scientist*

Recreate the legendary Sorting Hat using data science and machine learning magic!
This project implements a **multi-class logistic regression** model from scratch to classify Hogwarts students into their respective houses.

---

## 🎯 Project Overview

Professor McGonagall needs your help — the Sorting Hat has stopped working!
Your mission is to build a new one using pure muggle data science.

### Core Steps

* **Data Analysis:** Compute basic statistics manually (`count`, `mean`, `std`, `min`, `max`, etc.) without using helper functions like `pandas.describe()`.
* **Data Visualization:** Create meaningful plots to explore data:

  * Histogram (`histogram.py`)
  * Scatter plot (`scatter_plot.py`)
  * Pair plot (`pair_plot.py`)
* **Model Training:** Implement logistic regression (one-vs-all) using **gradient descent** from scratch.
* **Prediction:** Use your trained weights to predict student houses in `houses.csv`.

---

## 🧠 Learning Objectives

* Understand data preprocessing and visualization.
* Implement logistic regression manually.
* Grasp the mathematical foundation behind gradient descent and loss optimization.

---

## 🧩 Bonus Ideas

* Add more statistical metrics to your describe function.
* Try **stochastic** or **mini-batch gradient descent**.
* Implement different optimization algorithms.

---

## 🧾 Output Example

```
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Slytherin
```

Your goal: achieve at least **98% accuracy** — only then will Professor McGonagall accept your model as the true Sorting Hat.

---

## ⚙️ Technologies

* Python (recommended)
* NumPy, Matplotlib, Pandas (for visualization only)
* No machine learning libraries allowed (e.g., Scikit-Learn for training)

---

## 🪄 License

This project is for educational purposes and inspired by the *Harry Potter and a Data Scientist* exercise.
