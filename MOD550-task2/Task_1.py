'''
Project 1 for MOD550 by Bård Garathun
Uploaded for easier use of downloading the data to task 2
Divided up in tasks.
Task 3: make a histogram for the source data you selected
Task 4: convert the histogram into a discrete PMF
Task 5: calculate the cumulative for each feature
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Always change working directory to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DataFromSource:
    '''
    Load, filter, and visualize Eurostat/EEA emissions data.

    This class reads the CSV once at construction and exposes helpers to:
    - fetch a single location's time series,
    - plot/return a histogram,
    - convert that histogram to a discrete PMF and plot it,
    - compute/plot the CDF from the PMF.
    '''

    def __init__(self, country, bins):
        '''
        Initialize the class function
        '''
        self.country = country
        self.bins = bins

    def get_data(self):
        '''
        :param self.country: The name of the location (case sensitive).
        :param self.data_file: Path to the file holding the data.
        :print: Headers & unique locations.
        :return: A pandas datafram with emissions for the input location.
        '''
        df = pd.read_csv('data/emission.csv') 
        print(list(df)) # print the headers
        unique_locations = df['geo_label'].unique() # Find all unique locations in the dataframe
        print(unique_locations)
        try:
            df = df[df['geo_label'] == self.country]
        except:
            print(f'Could not find data for location {self.country}...')
            return None
        self.time =df['time'].to_numpy()
        self.emission = df['obs_value'].to_numpy()
        return self.time, self.emission

    def plot_histogram(self):
        '''
        Plots a histogram from input (task 3)
        Input:
        ----
        returned_data: Two dimensional list from func "random_numbers_in_two_dimensions(), use [0] for first row and [1] for second row.
        x: x_axis of returned_data
        y: y_axis of returned_data
        return: Plot
        '''
        #self.time, self.emission = self.get_data()
        plt.figure()
        plt.grid(True, alpha=0.3)
        plt.title(f'Emission distribution - {self.country}')
        plt.xlabel('Emission')
        plt.ylabel('Count')
        counts, edges, patches = plt.hist(self.emission, bins=self.bins, edgecolor='black')
        print('counts:', counts)
        print('bin_edges:', edges)
        plt.show()
        return counts, edges
    
    def plot_pmf(self):
        '''
        Task 4.
        Converting histogram into a discrete PMF
        '''
        counts, edges = np.histogram(self.emission, bins=self.bins, density=False)
        total = counts.sum()
        if total == 0:
            print("No data to build PMF.")
            return None
        pmf = counts / total
        centers = 0.5*(edges[:-1] + edges[1:])
        print('PMF\n', np.column_stack((centers, pmf)))
        plt.figure()
        markerline, stemlines, baseline = plt.stem(centers, pmf, basefmt=" ")
        plt.setp(markerline, marker='o', markersize=4)
        plt.title(f'Discrete PMF of Emissions – {self.country}')
        plt.xlabel('Emission (bin center)')
        plt.ylabel('Probability mass')
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        return centers, pmf
    
    def plot_cdf(self, centers, pmf):
        """
        Task 5: cumulative distribution from PMF
        """
        cdf = np.cumsum(pmf) # cumulative sum of an array
        print('CDF\n', np.column_stack((centers, cdf)))
        plt.figure()
        plt.plot(centers, cdf)
        plt.title(f'CDF of Emissions – {self.country}')
        plt.xlabel('Emission ')
        plt.ylabel('Cumulative probability')
        plt.ylim(0, 1.05)
        plt.grid()
        plt.show()
        return cdf
