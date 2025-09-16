import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Project 1 for MOD550 by Bård Garathun
Divided up in tasks.
Task 1: make a histogram from a 2d  random distribution
Task 2: make a 2d heat map from a 2d random distribution
Task 3: make a histogram for the source data you selected
Task 4: convert the histogram into a discrete PMF
Task 5: calculate the cumulative for each feature
'''

# Always change working directory to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class PracticePart:
    '''
    Class practicing task 1 ,2 and 3
    '''
    
    def __init__(self, x, y):
        '''
        Some init variables, will be filled in as they come
        '''
        self.x = x
        self.y = y


    def two_d_random_distribution(self,n_np, noise):
        '''
        Makes a random list with randomized noise. (task 1)
        Input:
        ----
        n_np: Specifies the list number
        noise: Noise in axis
        Return:
        ----
        Two dimentional list with randomized numbers in a positive trend
        '''
        self.x = np.random.rand(n_np)*10
        # y = ax+b
        true_slope = 10
        true_intercept = 10
        self.y = true_slope * self.x + true_intercept # Makes nice slope
        self.y += np.random.rand(n_np) * noise
        return self.x, self.y
    
    def plot_histogram(self, bins):
        '''
        Plots a histogram from input (task 3)
        Input:
        ----
        self.x: x_axis of returned_data
        self.y: y_axis of returned_data
        returns: histogram plot

        '''

        plt.grid()
        plt.title('Some random numbers')
        plt.xlabel('X-axis of random data')
        plt.ylabel('Y-axis of random data')
        #plt.hist(y, bins=10, edgecolor='blue') # Bins are the amount of bars you want to sort the data into
        plt.hist(self.x,bins=bins, edgecolor='black') # Bins are the amount of bars you want to sort the data into

        plt.show()

    def plot_heatmap(self, bins, cmap):
        '''
        plot a heatmap (task 2)
        Input:
        ----
        self.x: x_axis of returned_data
        self.y: y_axis of returned_data

        returns: Plot
        '''

        # Build 2D histogram (H is the grid), and edges define the bin boundaries
        H, xedges, yedges = np.histogram2d(self.x, self.y, bins=bins)

        # Plot with correct orientation and coordinate extent
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] # extent is a Matplotlib imshow kwarg: [xmin, xmax, ymin, ymax
        plt.figure(figsize=(6,5))
        plt.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap=cmap) # H.T transpose of H
        # histogram2d arranges H as H[ix, iy] where ix indexes x bins and iy indexes y bins.
        # origin='lower' flips that so y increases upward, like standard math/plotting.
        plt.colorbar(label='Count')
        plt.title('2D Heat Map of (X, Y)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(False)
        plt.show()



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
        returns: Plot
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
        plt.xlabel('Emission (bin center)')
        plt.ylabel('Cumulative probability')
        plt.ylim(0, 1.05)
        plt.grid()
        plt.show()
        return cdf

# Calling the practice part 1 and 2, keeping with the teachers cake naming conventions, due to it being a practice cake.
cake = PracticePart(x=1, y=1)
returned_data = cake.two_d_random_distribution(n_np = 2000, noise = 40)
cake.plot_heatmap(bins = 60, cmap='hot')
cake.plot_histogram(bins = 60)

# Calling the different functions from DataFromSource class on emission from new cars
EmClass = DataFromSource(country ='Belgium',bins = 'auto')
data = EmClass.get_data()
histo = EmClass.plot_histogram()
pmf_centers, pmf_pmf = EmClass.plot_pmf()
cumulative = EmClass.plot_cdf(pmf_centers, pmf_pmf)
