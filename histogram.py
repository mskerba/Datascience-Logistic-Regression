import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/dataset_train.csv")

sns.histplot(data=df, x='Care of Magical Creatures', hue="Hogwarts House", element="step")
plt.title(f"Distribution of Care of Magical Creatures by House")
plt.show()
