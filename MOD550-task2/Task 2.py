'''
Make a folder in your git MOD550-task2
This is going to be a big assignment but it is the last from my part. You have one month of time to work on it.
Inside the folder, put a Jupyter Notebook file  (.ipynb) for your Instances (cake) that you use to solve the task and one python file (.py) for your METHODS (recipes).
note from task 1 discussion: If you have a lot of data, please add the necessary only. You can have a single folder in your repo called data and you can access it from each task. (e.g. if your code sits in the folder MOD550-task2, you can access to your data naming the different folder such as: filename = "../data/a_lot_of_number.csv"). You wont need your data for now on task2, but you will need it later on.
 
task 2.1: Make a DataModel class that reads the output of the DataAquisition class (from task1) in its __init__()

task 2.2: Make a function in DataModel to make a linear regression. I suggest you try to do it on your own with only 
vanilla python and the class notes. If you are lost, you can find practical info here on how to use already 
made libraries:  https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/Lenker til en ekstern side.  
The issue here will be data structure: np.array vs list of list vs pandas DataFrames.
Can get help from Data_generation_and_classes in MOD550-fall-2025/notebooks

task 2.3: Make a function that split the data you got from DataAquisition into train, validation and test. Do it  with vanilla
 python. You need to make sure you understand the data structure.

task 2.4: Make a function that computes MSE (make your own, don't copy from my notes :P )

task 2.5: Make a function to make NN. It would be essentially a wrapper of other libraries, I suggest to use Keras: 
 https://www.geeksforgeeks.org/machine-learning/how-to-create-models-in-keras/  . You should have acquired enough notions 
 to handle this tool.

task 2.6: Make a function that does K_MEAN and GMM (we will discuss them next week)
Once these methods (recipes) are done, you can now make cakes! :) :

task 2.7: Make a linear regression on all your data (statistic ).

task 2.8: Make a linear regression on all your train data and test it on your validation.

task 2.9: Compute the MSE on your validation data. 

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
import sys
import os
import random
import sklearn.metrics as sk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
#from sklearn.linear_model import LinearRegression

# Get the absolute path to MOD550-task1
task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../MOD550-task1'))
print("Adding to sys.path:", task1_path)  # Debug print
sys.path.append(task1_path)
from mod550_project_1 import DataFromSource

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
        x, y = emission_class.get_data()
        self.x = x
        self.y = y
        print(x)

    def linear_regression(self):
        '''
        Task 2.2:
        Makes a linear regression from the input data in vanilla python

        :return a: calculated slope
        :return b: calculated intercept
        :return y_pred: predicted y values
        '''
        n = len(self.x)
        sum_x = sum(self.x) #np.sum(self.x)
        sum_y = sum(self.y) # np.sum(self.y)
        sum_xx = sum(x * x for x in self.x) # np.sum(self.x**2)
        sum_xy = sum(x * y for x, y in zip(self.x, self.y)) # np.sum(self.x*self.y)

        # Calculate the slope (a)
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        # Calculate the intercept (b)
        b = (sum_y - a * sum_x) / n
        # Predict y values
        y_pred = [a * x + b for x in self.x]
        print('a =', a, 'b =', b, 'x =', self.x)
        print('y_pred =', y_pred)
        return a, b, y_pred

        #Numpy implementation:
        # # Calculate the slope (a)
        # a = (n * np.sum(self.x*self.y) - np.sum(self.x)*np.sum(self.y)) / (n*np.sum(self.x**2) - (np.sum(self.x))**2)
        # # Calculate the intercept (b)
        # b = (np.sum(self.y) - a*np.sum(self.x)) / n
        # # Predict y values
        # y_pred = a*self.x + b
        # print('a =', a, 'b =', b, 'x =', self.x)
        # print(y_pred)
        # return a, b, y_pred

    def split_data(self, train = 0.5, val = 0.2, test = 0.3, seed=42):
        '''
        Task 2.3:
        Split the data you got from DataAquisition into train, validation and test. Using vanilla python

        :param train: Portion of data used for training
        :param val: Portion of data used for validation
        :param test: Portion of data used for testing
        :return x_train, y_train, x_val, y_val, x_test, y_test: X and Y values for training, validation and testing

        '''
        assert abs(train + val + test - 1.0) < 1e-4, "The sum of all the input values must be 1, try again :)"
        n = len(self.x)
        indices = list(range(n))
        if seed is not None:
            random.seed(seed) # Tells the pseudo RNG where to start its sequence of numbers
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

    def neural_network(self):
        '''
        Task 2.5:
        Make a function to make Neural Network (NN) using Keras

        :param something:

        :return something amazing <3: 
        '''
        visible = Input(shape=(10,))
        hidden = Dense(2)(visible)
        model = Model(inputs=visible, outputs=hidden)

    def plot(self):
        '''
        Plotting values and predictions

        :param self.x: Time value in years
        :param self.y: Emission value in g CO2 per km
        :param y_pred: predicted y values
        :return plot: Returns a plot
        '''
        # plots
        #plt.figure(figsize=(15, 12))
        plt.suptitle(
            f'Mean Squared Error: {mse}',
            fontsize=10
        )
        plt.scatter(self.x, self.y, label ='All data')
        plt.plot(self.x, y_pred, color='red', label='Fitted line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

model = DataModel(country ='Belgium')
a, b, y_pred = model.linear_regression()
x_train, y_train, x_val, y_val, x_test, y_test = model.split_data()
print('x_train', x_train, 'y_train', y_train, 'x_val', x_val, 'y_val', y_val, 'x_test', x_test, 'y_test', y_test)
mse = model.mean_squared_error(predicted= y_pred)
print("Mean Squared Error:", mse)
# NN = model.neural_network()
model.plot()
