import pandas as pd
import numpy as np
import argparse

learning_rate = 0.01   
max_epoch = 10000


def cost_function(h, house,m):
    j = 0
    for hi, yi in zip(h, house):
        j += yi * np.log(hi) + (1 - yi) * np.log(1 - hi)
    return - j/m

def gradient_descent(matrix, h_y, m):
    gd = matrix.dot(h_y)
    return gd/m

def new_theta(thetas, learning_rate, gradient):
    return thetas - learning_rate * gradient



def binary_target(df, house_name, batch_size=32):
    local = df.copy()

    house = (local["Hogwarts House"] == house_name).astype(int).to_numpy()

    if 'Hogwarts House' in local.columns:
      local = local.drop(columns=['Hogwarts House'])

    # features -> numpy (+ bias)
    matrix = local.to_numpy()
    matrix_with_ones = np.hstack((np.ones((matrix.shape[0], 1)), matrix))


    thetas = np.zeros(matrix_with_ones.shape[1])
    m = matrix_with_ones.shape[0]
    prev_cost = float("inf")

    for epoch in range(max_epoch):
        # shuffle indices each epoch
        idx = np.arange(m)
        np.random.shuffle(idx)

        # iterate mini-batches
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            b = idx[start:end]
            Xb = matrix_with_ones[b]        # (B, n+1)
            yb = house[b]                   # (B,)

            # zb = Xb @ thetas                # (B,)
            zb = Xb.dot(thetas)
            hb = 1 / (1 + np.exp(-zb))
            hb = np.clip(hb, 1e-12, 1 - 1e-12)

            # grad for this mini-batch: (n+1,)
            # grad_b = (Xb.T @ (hb - yb)) / (end - start)
            grad_b = gradient_descent(Xb.T, (hb - yb), (end - start))
            

            # update
            thetas = thetas - learning_rate * grad_b

        # z_full = matrix_with_ones @ thetas
        z_full = matrix_with_ones.dot(thetas)
        h_full = 1 / (1 + np.exp(-z_full))
        h_full = np.clip(h_full, 1e-12, 1 - 1e-12)

        cost = cost_function(h_full, house, m)
        if round(prev_cost, 7) == round(cost, 7):
            break
        prev_cost = cost

    preds = (h_full >= 0.5).astype(int)
    acc = np.mean(preds == house)
    print(f"Accuracy ({house_name} vs all):", acc)
    return thetas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression")
    parser.add_argument("input_file", nargs="?", default="datasets/dataset_train.csv",
                        help="path to training CSV file (default: datasets/dataset_train.csv)")
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df = df.dropna()
    df = df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Astronomy"])
    
    if "Best Hand" in df.columns:
        unique_hands = df["Best Hand"].unique()
        mapping = {value: idx for idx, value in enumerate(unique_hands)}
        df["Best Hand"] = df["Best Hand"].map(mapping)

    num_cols = df.select_dtypes(include=[np.number]).columns
    mu = df[num_cols].mean()
    sigma = df[num_cols].std().replace(0, 1.0) 
    df[num_cols] = (df[num_cols] - mu) / sigma

    
    thetas1 = binary_target(df, 'Ravenclaw')
    thetas2 = binary_target(df, 'Slytherin')
    thetas3 = binary_target(df, 'Gryffindor')
    thetas4 = binary_target(df, 'Hufflepuff')
    print("Thetas for Ravenclaw: ", thetas1)
    print("Thetas for Slytherin: ", thetas2)
    print("Thetas for Gryffindor: ", thetas3)
    print("Thetas for Hufflepuff: ", thetas4)

    thetas_df = pd.DataFrame({
        'Ravenclaw': thetas1,
        'Slytherin': thetas2,
        'Gryffindor': thetas3,
        'Hufflepuff': thetas4
    })

    # Save to CSV file
    thetas_df.to_csv("thetas-mini.csv", index=False)
    print("All theta values saved to thetas.csv")
