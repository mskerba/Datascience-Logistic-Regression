# pair_plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/dataset_train.csv")
df = df.drop(columns=[c for c in df.columns if c.lower() == "index" or "unnamed" in c.lower()], errors="ignore")
df = df.reset_index(drop=True)


features = df.select_dtypes(include="number").columns.tolist()
if not features:
    raise ValueError("No numeric columns found to plot.")


MAX_ROWS = 1500
plot_df = df.sample(n=MAX_ROWS, random_state=0) if len(df) > MAX_ROWS else df


sns.pairplot(
    plot_df,
    vars=features,
    hue="Hogwarts House",
    diag_kind="hist",
    corner=False
)

plt.suptitle("Pair Plot of Hogwarts Courses", y=1.02)
plt.tight_layout()
plt.show()
