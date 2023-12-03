# Mihir Singh and Jake Gilbert
# Handles Data Processing from Kaggle

# imports
import pandas as pd
import numpy as np

# the data processing class will be an object we use to handle data in a main file
class dataProcessing:
    # init function
    def __init__(self, fileInput):
        self.file = fileInput

        #info about data set --> can be easily changed if data being read in changes
        self.endDiscreteValues = 5
        self.endNonDiscreteValues = 9

    # reads data from Kaggle and puts it into a dataframe using pandas
    def importData(self):
        df = pd.read_csv(self.file)
        return df
    
    # cleans data by creating a list object where each day has characteristics associated with it
    # also takes input for how we clean data --> needs to be different if doing binary tree or something else
    def cleanData(self, df, setting):
        listedData = df.values.tolist()
        
        # cleaning for decision tree
        if(setting == 1):
            nonDiscretizedData = {}
            discretezedDatadelimeters = {}
            # want to find median, lower quartile, upper quartile for some of the data (temp, atemp, humidity, windspeed)
          
            #for i in range(listedData.size()):
            #   for j in range(self.endDiscreteValues, self.endNonDiscreteValues):
            #        nonDiscretizedData[j-self.endDiscreteValues] += listedData[i][j]
            #for k in range(self.endDiscreteValues, self.endNonDiscreteValues):
            #    for 

