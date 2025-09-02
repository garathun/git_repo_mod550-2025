import random

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
        print(list_of_numbers)
        return list_of_numbers
    
    def random_numbers_in_two_dimensions(self, number_of_randoms):
        


cake = MyFirstClass(input1=42)
cake.random_numbers(number_of_randoms=15)
cake.random_numbers_in_two_dimensions(number_of_randoms=10)

# First make a list of one dimensions of random numbers
# Homework, make a list of two dimension random number

