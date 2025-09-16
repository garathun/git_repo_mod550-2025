import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Project for MOD550
Divided up in tasks.
Task 1: make a histogram from a 2d  random distribution
Task 2: make a 2d heat map from a 2d random distribution
Task 3: make a histogram for the source data you selected
Task 4: convert the histogram into a discrete PMF
Task 5: calculate the cumulative for each feature
'''

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
        returned_data: Two dimensional list from func "random_numbers_in_two_dimensions(), use [0] for first row and [1] for second row.
        x: x_axis of returned_data
        y: y_axis of returned_data
        returns: Plot

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
        Task 2
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

# cake = PracticePart(x=1, y=1)
# returned_data = cake.two_d_random_distribution(n_np = 2000, noise = 40)
# cake.plot_heatmap(bins = 60, cmap='hot')
# cake.plot_histogram(bins = 60)

class DataFromSource:
    '''
    
    '''

    def __init__(self):
        '''
        
        '''
        #code

    def get_data(self, country):
        '''
        :param country: The name of the location (case sensitive).
        :param data_file: Path to the file holding the data.
        :print: Headers & unique locations.
        :return: A pandas datafram with the COVID-19 data for the input location.
        '''
        location = r'C:\Users\BårdGarathun\Documents\UiS\Classes\Mod550\Project\Project alternatives\eea_s_eu-sdg-13-31_p_2000-2023_v01_r00 (1)\eea_s_eu-sdg-13-31_p_2000-2023_v01_r00' 
        # File NAME, Must have \ at the beginning of the file name
        file_name = r'\eea_s_eu-sdg-13-31_p_2000-2023_v01_r00.csv'
        df = pd.read_csv(f"{location}{file_name}", sep=',')
        print(list(df)) # print the headers
        unique_locations = df['geo_label'].unique() # Find all unique places in the dataframe
        print(unique_locations)
        try:
            df = df[df['geo_label'] == country]
        except:
            print(f'Could not find data for location {country}...')
            return None
        time =df['time'].to_numpy()
        emission = df['obs_value'].to_numpy()
        return time, emission
    
    def plotting(self, country):
        '''
        :param country: The name of the location
        :param time: which year the recording is from
        :param emission: emission from the cars in 
        :return: Prints a plot showing
        '''
        time, emission = self.get_data(country)
        plt.subplot()
        plt.grid()
        plt.title(f'Emission in {country}')
        plt.xlabel('Year')
        plt.ylabel('Emission')
        plt.scatter(time, emission, color = 'black')
        plt.show()
    # def plot_histogram(self, country):
    #     '''
    #     Plots a histogram from input (task 3)
    #     Input:
    #     ----
    #     returned_data: Two dimensional list from func "random_numbers_in_two_dimensions(), use [0] for first row and [1] for second row.
    #     x: x_axis of returned_data
    #     y: y_axis of returned_data
    #     returns: Plot

    #     '''
    #     time, emission = self.get_data(country)
    #     # Define specific x and y values

    #     #plt.bar(time, emission, color='blue')
    #     plt.grid()
    #     plt.title('Some random numbers')
    #     plt.xlabel('Year')
    #     plt.ylabel('Emission')
    #     counts, edges, patches= plt.hist(emission, edgecolor='black') # Bins are the amount of bars you want to sort the data into
    #     print('counts = ', counts, 'bin_edges', edges, 'patches = ', patches)
    #     plt.show()

    #     return counts, edges
    def plot_histogram(self, country, bins='auto'):
        time, emission = self.get_data(country)
        plt.figure()
        plt.grid(True, alpha=0.3)
        plt.title(f'Emission distribution - {country}')
        plt.xlabel('Emission')
        plt.ylabel('Count')
        counts, edges, _ = plt.hist(emission, bins=bins, edgecolor='black')
        plt.show()
        return counts, edges
    
    def plot_pmf(self, country, bins='auto'):
        time, emission = self.get_data(country)
        counts, edges = np.histogram(emission, bins=bins, density=False)
        total = counts.sum()
        if total == 0:
            print("No data to build PMF.")
            return None, None
        pmf = counts / total
        centers = 0.5*(edges[:-1] + edges[1:])
        plt.figure()
        markerline, stemlines, baseline = plt.stem(centers, pmf, basefmt=" ")
        plt.setp(markerline, marker='o', markersize=4)
        plt.title(f'Discrete PMF of Emissions – {country}')
        plt.xlabel('Emission (bin center)')
        plt.ylabel('Probability mass')
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        return centers, pmf
    
    # def discrete_pmf():
    #     '''
    #     Converting histogram into a discrete PMF
    #     '''

    def plot_cdf(self, centers, pmf, country):
        """
        Task 5: cumulative distribution from PMF
        """
        cdf = np.cumsum(pmf)

        plt.figure()
        plt.plot(centers, cdf)
        plt.title(f'CDF of Emissions – {country}')
        plt.xlabel('Emission (bin center)')
        plt.ylabel('Cumulative probability')
        plt.ylim(0, 1.05)
        plt.grid()
        plt.show()
        return cdf

cake2 = DataFromSource()
cake1 = cake2.plotting(country='Belgium')
centers, pmf = cake2.plot_pmf(country='Belgium')
data2 = cake2.plot_histogram(country='Belgium')
cdf = cake2.plot_cdf(centers, pmf, country='Belgium')

'''
E means Estimated value.

BE (lowercase in Eurostat dictionaries: “be”) means Break in time series, estimated.

These are the standard Eurostat/EEA observation-status (flag) codes. “E” marks an observation that has been estimated;
 “be” (often shown uppercase as BE in some systems) combines two flags: “b” = break in time series and 
 “e” = estimated.
'''