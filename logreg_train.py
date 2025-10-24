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

def gradient_descent(matrix, h, house, m):
    h_y = h - house
    gd = matrix.dot(h_y)
    return gd/m

def new_theta(thetas, learning_rate, gradient):
    return thetas - learning_rate * gradient



def binary_target(df, house_name):
    house = (df["Hogwarts House"] == house_name).astype(int)
    if 'Hogwarts House' in df.columns:
        df = df.drop(columns=['Hogwarts House'])


    
    # Convert to numpy matrix
    matrix = df.to_numpy()
    # Add a column of ones to the left of the matrix
    matrix_with_ones = np.hstack((np.ones((matrix.shape[0], 1)), matrix))
    # matrix_with_ones = matrix_with_ones.astype(float) 


    # create thetas, initialized to zeros
    thetas = np.zeros(matrix_with_ones.shape[1])

    prev_cost = float("inf")


    for epoch in range(max_epoch):
        z = np.dot(matrix_with_ones, thetas)
  
        h = 1 / (1 + np.exp(-z))
        h = np.clip(h, 1e-12, 1 - 1e-12)
        

        cost = cost_function(h, house, len(matrix_with_ones))

        matrix_T = matrix_with_ones.T

        gradient_descent_r = gradient_descent(matrix_T, h, house, len(matrix_with_ones))

        thetas = new_theta(thetas, learning_rate, gradient_descent_r)
    
        if round(prev_cost, 7) == round(cost, 7):
            break
        prev_cost = cost


    predictions = (h >= 0.5).astype(int)
    accuracy = np.mean(predictions == house)
    print("Accuracy:", accuracy)
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
    thetas_df.to_csv("thetas.csv", index=False)
    print("All theta values saved to thetas.csv")
