import random
import numpy as np
import matplotlib.pyplot as plt


class MyFirstClass:
    '''
    Simple class just for the purpose of constructing classes
    '''
    def __init__(self, input1):
        '''
        Initialize the class function
        Input
        -----
        input1: Some input variable
        '''
        print('Some')
        self.input1 = input1

    def random_numbers(self,number_of_randoms):
        '''
        Makes a list of random numbers
        Input
        ----
        number_of_randoms: Specifies the number of random values
        list_of_numbers: The complete list
        '''
        list_of_numbers = []
        for i in range(number_of_randoms):
            list_of_numbers.append(random.randint(1,10))
        print(f'Just a list of numbers \n {list_of_numbers}')
        return list_of_numbers
    
    def random_numbers_in_two_dimensions(self, matrix_row, matrix_col):
        '''
        Makes a 2d list (matrix)
        Input
        ----
        matrix_row: Row numbers for the matrix
        matrix_col: Col numbers for the matrix
        :return two_dim_list: 2 dimentional matrix
        '''
        two_dim_list = []
        rows, cols = matrix_row, matrix_col
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(random.randint(1,10))
            two_dim_list.append(row)
        print('Beautifull 2d list aka matrix')

        ### Teachers solution
        n_p= 15
        list_xy = []
        for i in range(n_p):
            list_xy.append([random.random(), random.random()])
        print(list_xy)
        

        list_x, list_y, = [], []
        for i in range(n_p):
            list_x.append(random.random())
            list_y.append(random.random())
        # for row in two_dim_list:
            # print(row)

        # self.list_xy = list_xy
        self.list_xy = [list_x, list_y]
        return two_dim_list
    def ordered_list_generator(self, n_np, noise_x, noise_y):
        '''
        Makes a random list with randomized noise.
        Input:
        ----
        n_np: Specifies the list number
        noise_x: Noise in x-axis
        noise_y: Noise in y-axis
        Return:
        ----
        Two dimentional list with randomized numbers in a positive trend
        '''
        self.list_x, self.list_y = [], []
        for i in range(n_np):
            self.list_x.append(float(i)+random.uniform(0.0,noise_x))
            self.list_y.append(float(i)+random.uniform(0.0,noise_y))
            # y=ax+b
        return self.list_x, self.list_y
    def ordered_list_generation_v2(self,n_np, noise):
        '''
        Makes a random list with randomized noise.
        Input:
        ----
        n_np: Specifies the list number
        noise: Noise in axis
        Return:
        ----
        Two dimentional list with randomized numbers in a positive trend
        '''
        x = np.random.rand(n_np)
        # y = ax+b
        a = 10
        b = 10
        y = a*x+b # Makes nice slope
        y += np.random.rand(n_np)*noise
        self.list_x, self.list_y = [], []
        for i in range(n_np):
            self.list_x.append(x)
            self.list_y.append(y)
        return self.list_x, self.list_y


    def plotting(self,returned_data):
        '''
        Plots a scatter plot from input
        Input:
        ----
        returned_data: Two dimensional list from func "random_numbers_in_two_dimensions(), use [0] for first row and [1] for second row.
        return: Plot

        '''
        x = returned_data[0]
        y = returned_data[1]
        print(x)
        print(y)
        plt.subplot()
        plt.grid()
        plt.title('Some random numbers')
        plt.xlabel('X-axis of random data')
        plt.ylabel('Y-axis of random data')
        # plt.scatter(x,y)
        print('return')
        # print(self.list_xy)
        print(self.list_x)
        print(self.list_y)
        plt.scatter(self.list_x, self.list_y)
        # plt.scatter(self.list_xy[0],self.list_xy[1])

        # plt.scatter(*self.list_xy) # Also an alternative
        plt.show()



cake = MyFirstClass(input1=42)
# cake.random_numbers(number_of_randoms=5)
# returned_data =cake.random_numbers_in_two_dimensions(matrix_row=2, matrix_col=5)
# returned_data = cake.ordered_list_generator(n_np=50, noise_x=30, noise_y=30)
returned_data = cake.ordered_list_generation_v2(n_np = 30, noise = 5)
cake.plotting(returned_data)




# First make a list of one dimensions of random numbers
# Homework, make a list of two dimension random number
# Dont use pandas as that learns you syntac, not what you do

# Data generation procedure

# Assignment for next week
# He will share his code, I can use my own or whatever.
# Make a complete random number to a randomised with a trend
# You can use the cake.ordered_generator
# Cake = DataGenerator(100, distribution='random')
# Make a correlation graph.
# Best task is to have with random distribution