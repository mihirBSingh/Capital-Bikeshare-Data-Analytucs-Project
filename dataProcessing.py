# Mihir Singh and Jake Gilbert
# Handles Data Processing from Kaggle

# imports
import pandas as pd
import numpy as np

# the data processing class will be an object we use to handle data in a main file
class dataProcessing:
    def __init__(self):
        self.treeDelimiters = []

    # reads data from Kaggle and puts it into a dataframe using pandas
    def importData(self, fileInput):
        df = pd.read_csv(fileInput)
        return df       
    
    # collects delimiters used for classification
    def fileInfo(self, df):
        # collect nondiscretized data (temp, atemp, humidity, windspeed)
        tempVals = np.sort(df["temp"])
        atempVals = np.sort(df["atemp"])
        humidityVals = np.sort(df["humidity"])
        windspeedVals = np.sort(df["windspeed"])
        countVals = np.sort(df["count"])

        # want to find median, lower quartile, upper quartile for the data 
        tempDelimiter = {}
        tempDelimiter[0] = np.percentile(tempVals, 25)
        tempDelimiter[1] = np.percentile(tempVals, 50)
        tempDelimiter[2] = np.percentile(tempVals, 75)
        atempDelimiter = {}
        atempDelimiter[0] = np.percentile(atempVals, 25)
        atempDelimiter[1] = np.percentile(atempVals, 50)
        atempDelimiter[2] = np.percentile(atempVals, 75)
        humidityDelimiter = {}
        humidityDelimiter[0] = np.percentile(humidityVals, 25)
        humidityDelimiter[1] = np.percentile(humidityVals, 50)
        humidityDelimiter[2] = np.percentile(humidityVals, 75)
        windspeedDelimiter = {}
        windspeedDelimiter[0] = np.percentile(windspeedVals, 25)
        windspeedDelimiter[1] = np.percentile(windspeedVals, 50)
        windspeedDelimiter[2] = np.percentile(windspeedVals, 75)
        countDelimiter = {}
        countDelimiter[0] = np.percentile(countVals, 25)
        countDelimiter[1] = np.percentile(countVals, 50)
        countDelimiter[2] = np.percentile(countVals, 75)
        
        self.treeDelimiters.append(tempDelimiter)
        self.treeDelimiters.append(atempDelimiter)
        self.treeDelimiters.append(humidityDelimiter)
        self.treeDelimiters.append(windspeedDelimiter)
        self.treeDelimiters.append(countDelimiter)
        
    # cleans data by creating a list object where each day has characteristics associated with it
    # also takes input for how we clean data --> needs to be different if doing binary tree or something else
    def cleanData(self, df):
            
        # always train model first
        tempDelimiter = self.treeDelimiters[0]
        atempDelimiter = self.treeDelimiters[1]  
        humidityDelimiter = self.treeDelimiters[2]
        windspeedDelimiter = self.treeDelimiters[3]  
        countDelimiter = self.treeDelimiters[4]             

        # with our delimiters collected in fileInfoForTree we can discretize the data
        df.loc[df["temp"] <= tempDelimiter[0], "temp"] =  0
        df.loc[(df["temp"] > tempDelimiter[0]) & (df["temp"] <= tempDelimiter[1]), "temp"] =  1
        df.loc[(df["temp"] > tempDelimiter[1]) & (df["temp"] <= tempDelimiter[2]), "temp"] =  2
        df.loc[df["temp"] > tempDelimiter[2], "temp"] =  3

        df.loc[df["atemp"] <= atempDelimiter[0], "atemp"] =  0
        df.loc[(df["atemp"] > atempDelimiter[0]) & (df["atemp"] <= atempDelimiter[1]), "atemp"] =  1
        df.loc[(df["atemp"] > atempDelimiter[1]) & (df["atemp"] <= atempDelimiter[2]), "atemp"] =  2
        df.loc[df["atemp"] > atempDelimiter[2], "atemp"] =  3

        df.loc[df["humidity"] <= humidityDelimiter[0], "humidity"] =  0
        df.loc[(df["humidity"] > humidityDelimiter[0]) & (df["humidity"] <= humidityDelimiter[1]), "humidity"] =  1
        df.loc[(df["humidity"] > humidityDelimiter[1]) & (df["humidity"] <= humidityDelimiter[2]), "humidity"] =  2
        df.loc[df["humidity"] > humidityDelimiter[2], "humidity"] =  3

        df.loc[df["windspeed"] <= windspeedDelimiter[0], "windspeed"] =  0
        df.loc[(df["windspeed"] > windspeedDelimiter[0]) & (df["windspeed"] <= windspeedDelimiter[1]), "windspeed"] =  1
        df.loc[(df["windspeed"] > windspeedDelimiter[1]) & (df["windspeed"] <= windspeedDelimiter[2]), "windspeed"] =  2
        df.loc[df["windspeed"] > windspeedDelimiter[2], "windspeed"] =  3
            
        df.loc[df["count"] <= countDelimiter[0], "count"] =  0
        df.loc[(df["count"] > countDelimiter[0]) & (df["count"] <= countDelimiter[1]), "count"] =  1
        df.loc[(df["count"] > countDelimiter[1]) & (df["count"] <= countDelimiter[2]), "count"] =  2
        df.loc[df["count"] > countDelimiter[2], "count"] =  3
            
        return df
            
            
          

