
'''
Task 2 for MOD550 by Bård Garathun

For task 2.7, 2.8 and 2.9, it looks like the dataset might cause a problem.
Either because I don't see how I should handle the data to make the best use of it,
or because it's too fragmented and the different fragments have a max lenght of 23.
It give a hailstorm of lines when I do linear regression on each country to plot it together in a graph.
But I cannot combine as that will just give several y values for each x value and therefore no added value.
Can group together in larger categories than country, like east, west, north, south europe or similar,
get mean/avg or other and use this to compare reginal differences? Or find a way to calculate the neighbouring difference
based on location and distance.
Anyway, I'm open for suggestions if I can use this dataset or if I should consider something else.



task 2.10: Try for different distribution of initial data point, (a) Discuss how different functions can be used in the 
linear regression, and different NN architecture. (b) Discuss how you can use the validation data for the different cases. 
(c) Discuss the different outcome from the different models when using the full dataset to train and when you use a different
 ML approach. (d) Discuss the outcomes you get for K-means and GMM. (e) Discuss how you can integrate supervised and
 unsupervised methods for your case.
 
Comment:  All previous tasks are made to make you arrive to do 2.10. This point is the core of the course and it is what you 
need to learn in general when you work with ML and in particular if you want to pass the exam :P 
 
PREPARE FOR BUT NOT DUE YET:  task 3: apply all this to your dataset (this will be part of the final task) 
To submit, load on git your work and REPLY here with a comment: (e.g. 'Done)
'''
# import sys
# import os
import random
# import sklearn.metrics as sk
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Input
from keras.layers import Dense
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
# from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Task_1 import DataFromSource

class DataModel:
    '''
    Complete DataModel class for task 2.
    '''
    def __init__(self, country):
        '''
        Task 2.1.
        DataModel class that reads the output of the DataAquisition class (from task1)

        :param self.x: Time value in years
        :param self.y: Emission value in g CO2 per km
        '''
        
        emission_class = DataFromSource(country =country,bins = 'auto')
        x, y, unique_locations = emission_class.get_data()
        self.x = x
        self.y = y
        self.unique_locations = unique_locations

    def linear_regression(self, x, y):
        '''
        Task 2.2:
        Makes a linear regression from the input data in vanilla python
        Added param x, y to make use of same function in task 2.8 and 2.9

        :param x: x_train values used for training
        :param y: y_train values used for training
        :return a: calculated slope
        :return b: calculated intercept
        :return y_pred: predicted y values
        '''

        if x == None:
            x = self.x
        if y == None:
            y = self.y
        n = len(x)
        if n < 2:
            print(f'Not enough points({n}) to make a regression')
            a = 0
            b = 0
            y_pred = [0]*n
            return a, b, y_pred
        sum_x = sum(x) #np.sum(self.x)
        sum_y = sum(y) # np.sum(self.y)
        sum_xx = sum(x * x for x in x) # np.sum(self.x**2)
        sum_xy = sum(x * y for x, y in zip(x, y)) # np.sum(self.x*self.y)
        # Calculate the slope (a)
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        # Calculate the intercept (b)
        b = (sum_y - a * sum_x) / n
        # Predict y values
        y_pred = [a * x + b for x in x]
        print('a =', a, 'b =', b, 'x =', x)

        return a, b, y_pred
    
        #Numpy implementation: # Keeping for future use
        # # Calculate the slope (a)
        # a = (n * np.sum(self.x*self.y) - np.sum(self.x)*np.sum(self.y)) / (n*np.sum(self.x**2) - (np.sum(self.x))**2)
        # # Calculate the intercept (b)
        # b = (np.sum(self.y) - a*np.sum(self.x)) / n
        # # Predict y values
        # y_pred = a*self.x + b
        # print('a =', a, 'b =', b, 'x =', self.x)
        # print(y_pred)
        # return a, b, y_pred

    def split_data(self, train = 0.6, val = 0.2, test = 0.2, RSEED=None):
        '''
        Task 2.3:
        Split the data you got from DataAquisition into train, validation and test. Using vanilla python

        :param train: Portion of data used for training
        :param val: Portion of data used for validation
        :param test: Portion of data used for testing
        :param RSEED: Sets start value for pseudo RNG 
        :return x_train, y_train, x_val, y_val, x_test, y_test: X and Y values for training, validation and testing
        '''

        assert abs(train + val + test - 1.0) < 1e-4, "The sum of all the input values must be 1, try again :)"
        n = len(self.y)
        indices = list(range(n))
        if RSEED is not None:
            random.seed(RSEED) # Tells the pseudo RNG where to start its sequence of numbers
            # If using default seed = None. It will give "true" random sequnce of numbers
        random.shuffle(indices) # Shuffles the "cards", randomly rearanges the indices
        train_end = int(train * n) # Calculates where to split
        val_end = train_end + int(val * n)
        train_idx = indices[:train_end] # From start to train_end
        val_idx = indices[train_end:val_end] # From train_end to val_end
        test_idx = indices[val_end:] # val_end to infinity and beyond
        x_train = [self.x[i] for i in train_idx] # Picks out the values that belong in training from the pre-randomized dataset
        y_train = [self.y[i] for i in train_idx] # Does not uses range, but list
        x_val = [self.x[i] for i in val_idx]
        y_val = [self.y[i] for i in val_idx]
        x_test = [self.x[i] for i in test_idx]
        y_test = [self.y[i] for i in test_idx]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def mean_squared_error(self, predicted):
        '''
        Task 2.4:
        Make a function that computes mean squared error (MSE)
        Formula for MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

        :param self.y: Observed values
        :param predicted: Predicted values
        :return mse: Mean squared error
        '''

        n = len(self.y)
        if n != len(predicted):
            raise ValueError('Predicted and real value list length must be the same')
        error = 0 # Initiate variable
        for i in range(n):
            error += (self.y[i] - predicted[i])**2
        mse = error / n
        return mse

    def neural_network(self, epochs):
        '''
        Task 2.5:
        Make a function to make Neural Network (NN) using Keras

        :param epochs: The number of training cycles
        :return y_pred_nn: The predicted values from the neural network
        :return history_nn: The training history object, useful for plotting loss
        '''

        # Reshape the data
        x_reshaped = self.x.reshape(-1, 1) # sklearn expects a 2D array for features
        y_reshaped = self.y.reshape(-1, 1) # sklearn expects a 2D array for features 
        # Scale the data between (0-1)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x_reshaped)
        y_scaled = scaler_y.fit_transform(y_reshaped)
        # Define the neural network model
        model_nn = Sequential([
            Dense(16, input_dim=1, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(32, activation= 'relu', kernel_regularizer = l2(0.001)),
            Dense(64, activation= 'relu', kernel_regularizer = l2(0.001)),
            Dense (1) # Output layer without activation for linear regression
            ])
        # Compile the model
        model_nn.compile(optimizer=Adam(learning_rate=5e-4), loss='mean_squared_error')
        # Train the model
        history_nn = model_nn.fit(x_scaled, y_scaled, epochs = epochs, verbose = 1)
        # Use the model for prediction
        y_pred_nn_scaled = model_nn.predict(x_scaled)
        # INVERSE TRANSFORM THE PREDICTION
        # This converts the prediction from the 0-1 scale back to the original emission scale
        y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled)
        return y_pred_nn, history_nn
    
    def k_mean(self, n_clusters, RSEED=None):
        '''
        Task 2.6: K_MEAN
        Make a function that does K_MEAN and GMM (we will discuss them next week)
        Performs K-Means clustering on the data

        :param n_clusters: The number of clusters (K) to find
        :param RSEED: The random seed for reproducibility
        :return labels: The cluster assignment for each data point
        :return model: The fitted KMeans model object
        '''

        X = np.array(list(zip(self.x, self.y)))
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Create and fit the K-Means model
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=RSEED, n_init=10)
        kmeans_model.fit(X_scaled)
        # Get the cluster labels for each data point
        labels = kmeans_model.labels_
        print(f"\nK-Means found {len(np.unique(labels))} clusters")
        return labels, kmeans_model
    
    def gmm(self, n_components, RSEED=None):
        '''
        Task 2.6: GMM
        Make a function that does K_MEAN and GMM

        :param n_components: The number of Gaussian distributions to model
        :param RSEED: The random seed for reproducibility
        :return labels: The cluster assignment for each data point
        :return model: The fitted GMM model object
        '''

        X = np.array(list(zip(self.x, self.y)))
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Create and fit the GMM
        gmm_model = GaussianMixture(n_components=n_components, random_state=RSEED)
        gmm_model.fit(X_scaled)
        # Get the cluster labels for each data point
        labels = gmm_model.predict(X_scaled)
        print(f"\nGMM found {len(np.unique(labels))} clusters")
        return labels, gmm_model
    
    def linear_regression_on_all_data(self):
        '''
        Task 2.7:
        Performs a linear regression on the data from each county
        Uses linear_regression to make regression for each country
        Uses plot_all_country_regressions for plotting for each country

        :return all_train_regression_results: A dictionary containing the regression results for each country
        x_train, y_train, y_pred_train, a_train, mse_val
        '''

        all_regression_results = {}

        for country in self.unique_locations:
            print(f"Processing: {country}")
            model = DataModel(country=country)
            if len(model.x) < 6:
                print(f'Skipping {country} due to insufficient data')
                continue
            a, b, y_pred = 0, 0, [] # Safety line for try block
            try:
                a, b, y_pred = model.linear_regression(x= None, y= None)
            except ZeroDivisionError as e:
                print(f'{e} for {country}')
            # Store the results (original data and the prediction) in a dictionary
            all_regression_results[country] = {
                'x': model.x,
                'y': model.y,
                'y_pred': y_pred,
                'slope': a
            }
        if all_regression_results:
            model.plot_all_country_regressions(all_regression_results)
    
    def LR_on_all_train_data(self,train, val, test, RSEED):
        '''
        Task 2.8 & 2.9
        Performs a linear regression on the data from each county on train data
        Calculates MSE for each country on validate data
        Uses linear_regression to make regression for each country
        Uses plot_all_country_regressions for plotting for each country

        :param RSEED: The random seed for reproducibility
        :return all_train_regression_results: A dictionary containing the regression results for each country
        x_train, y_train, y_pred_train, a_train, mse_val
        :return all_validation_results: A dictionary containing the MSE results for each country
        '''

        all_train_regression_results = {}
        all_validation_results = {}

        for country in self.unique_locations:
            print(f"Processing: {country}")
            model = DataModel(country=country)
            if len(model.x) < 6:
                print(f'Skipping {country} due to insufficient data')
                continue
            a_train, b_train, y_pred_train = 0, 0, [] # Safety line for try block
            mse_val = 0
            x_train, y_train, x_val, y_val, x_test, y_test = model.split_data(train = 0.6, val = 0.2, test = 0.2, RSEED=RSEED)
             
            try:
                a_train, b_train, y_pred_train = model.linear_regression(x_train, y_train)
            except ZeroDivisionError as e:
                print(f'{e} for {country}')
            if len(x_val) > 0:
                y_pred_validation = [a_train * x + b_train for x in x_val]
                # Calculate MSE
                mse_val = np.mean((np.array(y_val) - np.array(y_pred_validation))**2)
                print(f"Validation MSE for {country}: {mse_val:.4f}")
                # Store the result
                all_validation_results[country] = mse_val
            else:
                print(f"No validation data for {country} to calculate MSE.")
            # Store the results (original data and the prediction) in a dictionary
            all_train_regression_results[country] = {
                'x': x_train,
                'y': y_train,
                'y_pred': y_pred_train,
                'slope': a_train,
                'mse' : mse_val
            }
        if all_train_regression_results:
            model.plot_all_country_regressions(all_train_regression_results)


    def plot_neural_network(self):
        '''
        Plotting Model loss during training, original data, linear regression and NN prediction

        :param y_pred_nn: The predicted values from the neural network
        :param history_nn: The training history object, useful for plotting loss
        :return: Returns two plots
        '''

        plt.figure(figsize=(14, 6))
        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(nn_history.history['loss'])
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        # Plot 2: Model Predictions
        plt.subplot(1, 2, 2)
        plt.scatter(self.x, self.y, label='Original Data')
        plt.plot(self.x, y_pred, color='red', linewidth=2, label='Linear Regression')
        plt.plot(self.x, nn_y_pred, color='black', linestyle='--', linewidth=2, label='Neural Network Prediction')
        plt.title('Model Comparison')
        plt.xlabel('Year')
        plt.ylabel('Emissions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjusting plot to prevent labels from overlapping
        plt.show()

    def plot_clusters(self, labels, model_name):
        '''
        Task 2.6:
        Plots the data points, colored by their assigned cluster label

        :param labels: A list or array of cluster labels for each data point
        :param model_name: The name of the clustering model (K-Means or GMM)
        '''

        # Use the original, un-scaled data for plotting
        X_original = np.array(list(zip(self.x, self.y)))
        plt.figure(figsize=(10, 7))
        # Use a scatter plot, with the 'c' parameter set to the cluster labels
        plt.scatter(X_original[:, 0], X_original[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.title(f'Clusters Found by {model_name}')
        plt.xlabel('Year')
        plt.ylabel('Emissions (g CO2 per km)')
        plt.colorbar(label='Cluster ID')
        plt.grid(True)
        plt.show()

    def plot_all_country_regressions(self, results_dict):
        '''
        Task 2.7, 2.8 and 2.9:
        Plots the linear regression lines for multiple countries on a single graph

        TODO Figure out if checkboxes can be done in a live graph, prob use tkinter

        :param results_dict: A dictionary containing the regression results for each country
        x_train, y_train, y_pred_train, a_train, mse_val
        '''
        
        plt.figure(figsize=(14, 7))
        # Plots all the raw data points with some transparency
        for country, data in results_dict.items():
            plt.scatter(data['x'], data['y'], alpha=0.2, label=f'_{country} data') # Underscore hides from legend
        for country, data in results_dict.items():
            # The label includes the country, it's calculated slope and MSE for easy comparison
            try:
                plt.plot(data['x'], data['y_pred'], linewidth=2, label=f'{country} (Slope: {data["slope"]:.2f}, MSE {data["mse"]:.2f})')
            except KeyError:
                plt.plot(data['x'], data['y_pred'], linewidth=2, label=f'{country} (Slope: {data["slope"]:.2f})')
            except ValueError as e:
                print(f'Valueerror({e}) for {country}')
        plt.title('Linear Regression Comparison Across All Countries')
        plt.xlabel('Year')
        plt.ylabel('Emissions (g CO2 per km)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize= 8)
        plt.tight_layout(rect=[0, 0, 0.95, 1]) # Fitting ledend to the right side without dissapearing
        plt.grid(True, alpha=0.5)
        plt.show()



# The messy kitchen...

# Starts by setting the RSEED to either None or 100 for checking code 
RSEED = None

# Task 2.1
model = DataModel(country ='Sweden')

# Task 2.2
a, b, y_pred = model.linear_regression(x= None, y= None)
model.split_data(train = 0.6, val = 0.2, test = 0.2, RSEED= RSEED)

# Task 2.3
x_train, y_train, x_val, y_val, x_test, y_test = model.split_data(train = 0.6, val = 0.2, test = 0.2, RSEED= RSEED)
print('x_train', x_train, 'y_train', y_train, 'x_val', x_val, 'y_val', y_val, 'x_test', x_test, 'y_test', y_test)

# Task 2.4
mse = model.mean_squared_error(predicted= y_pred)
print("Mean Squared Error:", mse)

# Task 2.5
nn_y_pred, nn_history = model.neural_network(epochs=600)
model.plot_neural_network()

# Task 2.6
# K-Mean
clusters = 5
kmean_labels, kmean_model = model.k_mean(n_clusters=clusters, RSEED= RSEED)
model.plot_clusters(kmean_labels, f'K-Means (K={clusters})')
# GMM
gmm_labels, gmm_model = model.gmm(n_components=clusters, RSEED= RSEED)
model.plot_clusters(gmm_labels, f'GMM (N={clusters})')

# Task 2.7
model.linear_regression_on_all_data()

# Task 2.8 and 2.9
model.LR_on_all_train_data(train = 0.6, val = 0.2, test = 0.2, RSEED= RSEED)
