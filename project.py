#    CS 422 Group Project - Predicting Housing Prices using KNN, Linear
#       Regression, and SVM
#    @file project.py
#    @members Ryan Parker, Keith Del Rosario, Daniel Shina, Matthew Shiroma


"""All the imports that will be needed for all tasks"""
import statistics as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from prettytable import PrettyTable
import matplotlib.pyplot as plt

K_FOLD = 5 # fold value for all functions
K_NEAREST_NEIGHBORS = 10 # the # of neighbors for KNN

def knn_function(features, target):
    """Function that runs KNN and returns mean RMSE"""
    kf = KFold(n_splits=K_FOLD)

    # create a list to store the RMSE values for each fold
    rmse_scores = []

    # Loop through each fold
    for train_index, test_index in kf.split(features):
        # Split the data into training and testing sets for the current fold
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Create a KNN Regression model
        knn = KNeighborsRegressor(n_neighbors=K_NEAREST_NEIGHBORS)

        # Train the model on the training data
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Calculate the RMSE score for the fold
        rmse = st.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)

    return rmse_scores

def linear_regression(features, target):
    """Function performs linear regression on features and target data set"""
    kf = KFold(n_splits=K_FOLD)

    # Create a list to store the RMSE values for each fold
    rmse_scores = []

    # Loop through each fold
    for train_index, test_index in kf.split(features):
        # Split the data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Create a linear regression model and fit the data
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = lr.predict(X_test)

        # Calculate the RMSE for this fold and add it to the list of RMSE scores
        rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse_fold)

    return rmse_scores # return the rmse values

def svm_function(features, target):
    """This function performs svm on a features and target data set"""
    kf = KFold(n_splits=K_FOLD) # declare the kfold

    rmse_scores = [] # delcare empty array

    svm = SVR(kernel="rbf") # declare svm 

    # loop through every fold
    for train_index, test_index in kf.split(features):
        # Split the data into training and testing sets
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # fit the model
        svm.fit(X_train, y_train)

        # get the prediction
        y_pred = svm.predict(X_test)

        # get the rmse error 
        rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse_fold)

    return rmse_scores # return the rmse values

def table_printing(knn_scores, linear_scores, svm_scores):
    """Prints the table of outputs using the prettytable module"""
    print("RMSE Result Table for Housing Data Prediction:")

    # the header for the table
    my_table = PrettyTable(["", "KNN", "Linear", "SVM"])

    for row in range(K_FOLD): # goes through every row
        table_array = []
        fold_string = f"Fold {row+1}" # creates an iterated fold string
        table_array.append(fold_string)

        # fills in the column with the correct formatted value
        for col in range(3):
            if col == 0:
                table_array.append(f"{knn_scores[row]:.2f}")
            elif col == 1:
                table_array.append(f"{linear_scores[row]:.2f}")
            elif col == 2:
                table_array.append(f"{svm_scores[row]:.2f}")

        my_table.add_row(table_array) # push the row to the table

    # fills in the last row which is always average
    table_array = []
    table_array.append("Mean")
    table_array.append(f"{np.mean(knn_scores):.2f}")
    table_array.append(f"{np.mean(linear_scores):.2f}")
    table_array.append(f"{np.mean(svm_scores):.2f}")
    my_table.add_row(table_array)

    print(my_table) # prints the table

def main():
    """Main function which calls the other functions with the proper data"""
    data = pd.read_csv("kc_final.csv")

    # drops the outlying columns and price
    features = data.drop(["id","date","price", "waterfront", "view", "condition",
                          "grade", "zipcode", "lat", "long"], axis=1)
    features.drop(features.columns[[0]], axis=1, inplace=True) # drops the row number column

    target = data["price"]

    # call the functions for every model
    print("Running KNN")
    knn_scores = knn_function(features, target)

    print("Running Linear Regression")
    linear_scores = linear_regression(features, target)

    print("Running SVM with RBF kernel")
    svm_scores = svm_function(features, target)

    print("\n") # formatting

    # print the mean RMSE for each model
    print(f"KNN Mean RMSE is: {np.mean(knn_scores)}")
    print(f"Linear Regression Mean RMSE is: {np.mean(linear_scores)}")
    print(f"SVM Mean RMSE is: {np.mean(svm_scores)}")

    print("\n") # formatting

    # print out the table
    table_printing(knn_scores, linear_scores, svm_scores)

    # inserting a 0 into the beginning of arrays as fold x is index x-1
    knn_scores.insert(0,0)
    linear_scores.insert(0,0)
    svm_scores.insert(0,0)

    # print the plot
    plt.plot(knn_scores, label="KNN")
    plt.plot(linear_scores, label="Linear Regression")
    plt.plot(svm_scores, label="SVM")
    plt.legend(loc="best")
    plt.title("Plot of RMSE Values For Given Models")
    plt.xlabel("Folds")
    plt.ylabel("RMSE Value")
    plt.show()

if __name__ == "__main__":
    main()
