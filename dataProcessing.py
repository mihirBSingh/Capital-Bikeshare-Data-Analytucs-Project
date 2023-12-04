# Mihir Singh and Jake Gilbert
# Handles Data Processing from Kaggle

# imports
import pandas as pd
import numpy as np

# the data processing class will be an object we use to handle data in a main file
class dataProcessing:
    def __init__(self):
        self.treeDelimiters = {}

    # reads data from Kaggle and puts it into a dataframe using pandas
    def importData(self, fileInput):
        df = pd.read_csv(fileInput)
        return df       
    
    def fileInfoForTree(self, df):
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
        tempDelimiter[2] = np.åpercentile(tempVals, 75)
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
        
        delimiters = {tempDelimiter, atempDelimiter, humidityDelimiter, windspeedDelimiter, countDelimiter}
        return delimiters 
        
    # cleans data by creating a list object where each day has characteristics associated with it
    # also takes input for how we clean data --> needs to be different if doing binary tree or something else
    def cleanData(self, df, setting, train):
        listedData = df.values.tolist()
        
        # cleaning for decision tree
        if(setting == 1): 
            
            # collect delimiters if training --> if testing then we want same ones from training file so we don't collect (should be done/trained already)
            if(train):
                self.treeDelimiters = self.fileInfoFOrTree(self, df)
            
            # will throw error if we haven't trained --> always train model first
            tempDelimiter = self.treeDelimiters[0]
            atempDelimiter = self.treeDelimiters[1]  
            humidityDelimiter = self.treeDelimiters[2]
            windspeedDelimiter = self.treeDelimiters[3]  
            countDelimiter = self.treeDelimiters[4]             

            # with our delimiters collected in fileInfoForTree we can discretize the data
            df.loc[df["temp"] <= tempDelimiter[0]] =  0
            df.loc[df["temp"] > tempDelimiter[0] & df["temp"] <= tempDelimiter[1]] =  1
            df.loc[df["temp"] > tempDelimiter[1] & df["temp"] <= tempDelimiter[2]] =  2
            df.loc[df["temp"] > tempDelimiter[2]] =  3

            df.loc[df["atemp"] <= atempDelimiter[0]] =  0
            df.loc[df["atemp"] > atempDelimiter[0] & df["atemp"] <= atempDelimiter[1]] =  1
            df.loc[df["atemp"] > atempDelimiter[1] & df["atemp"] <= atempDelimiter[2]] =  2
            df.loc[df["atemp"] > atempDelimiter[2]] =  3

            df.loc[df["humidity"] <= humidityDelimiter[0]] =  0
            df.loc[df["humidity"] > humidityDelimiter[0] & df["humidity"] <= humidityDelimiter[1]] =  1
            df.loc[df["humidity"] > humidityDelimiter[1] & df["humidity"] <= humidityDelimiter[2]] =  2
            df.loc[df["humidity"] > humidityDelimiter[2]] =  3

            df.loc[df["windspeed"] <= windspeedDelimiter[0]] =  0
            df.loc[df["windspeed"] > windspeedDelimiter[0] & df["windspeed"] <= windspeedDelimiter[1]] =  1
            df.loc[df["windspeed"] > windspeedDelimiter[1] & df["windspeed"] <= windspeedDelimiter[2]] =  2
            df.loc[df["windspeed"] > windspeedDelimiter[2]] =  3
            
            df.loc[df["count"] <= countDelimiter[0]] =  0
            df.loc[df["count"] > countDelimiter[0] & df["count"] <= countDelimiter[1]] =  1
            df.loc[df["count"] > countDelimiter[1] & df["count"] <= countDelimiter[2]] =  2
            df.loc[df["count"] > countDelimiter[2]] =  3
            
        return df
            
            
          

