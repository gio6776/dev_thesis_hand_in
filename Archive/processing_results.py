import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class ProcessResults():
    '''
    Class to process the results.
    It contains functions to create visualizations and analyze the results.
    '''
    def __init__(self, results, sheet_name):
        # read excel file into a pandas dataframe and store it in the class
        self.df = pd.read_excel(results, sheet_name=sheet_name)
        # print all columns of the dataframe
        print(f'Entire DF:\n {self.df.columns}')

    def plot_pie_correctness(self):
        # Counting the occurrences of each error classification
        error_counts = self.df['error_classification'].value_counts().reset_index()
        error_counts.columns = ['error_classification', 'count']  # Rename columns to be more descriptive

        # Sort error_counts to ensure color mapping consistency, most frequent first
        error_counts.sort_values('count', ascending=False, inplace=True)

        # Define the color scale (from darkest to lightest red for the errors)
        color_scale = ['#B71C1C', '#EF5350', '#E57373', '#EF9A9A', '#FFCDD2']  # Darkest to lightest

        # Mapping errors to colors, excluding 'correct' which should always be green
        colors = []
        pull = []  # To control the pull-out effect
        correct_index = None  # Track index of 'correct' for proper color assignment
        for i, row in enumerate(error_counts.itertuples()):
            if row.error_classification == 'correct':
                colors.append('green')
                pull.append(0.1)  # Pull out the 'correct' slice
                correct_index = i
            else:
                # Assign colors based on order of frequency, not index directly
                color_index = i if correct_index is None else i - 1
                colors.append(color_scale[color_index % len(color_scale)])
                pull.append(0)  # No pull-out effect for errors other than 'correct'

        # Creating a new DataFrame for sorting legend appropriately
        legend_df = error_counts.copy()
        legend_df['color'] = colors
        legend_df['pull'] = pull

        # Sort legend: 'correct' first, followed by error classifications from light to dark
        legend_df.sort_values(by=['color', 'count'], ascending=[False, True], inplace=True, key=lambda x: x.replace('green', '#000000'))

        # Labels and values for the pie chart
        labels = legend_df['error_classification']
        values = legend_df['count']
        colors = legend_df['color']
        pull = legend_df['pull']

        # Constructing the pie chart using plotly.graph_objects
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, pull=pull)])
        fig.update_layout(title_text='Distribution of Error Classifications', margin=dict(l=20, r=20, t=20, b=20), legend_font=dict(size=14), legend_traceorder='reversed')

        # Showing the figure
        fig.show()




    def get_df(self, clean_df=True):
        '''
        Function to get the results dataframe
        '''
        if clean_df:
            # select only the columns needed for the analysis
            columns_needed = ['id','sql_query', 'user_question', 'complexity', 'correctness', 'error_classification']
            columns_present = [col for col in columns_needed if col in self.df.columns]
            self.df = self.df[columns_present]
            print(f'Cleaned DF:\n {self.df.columns}')
        else:
            pass

        return self.df