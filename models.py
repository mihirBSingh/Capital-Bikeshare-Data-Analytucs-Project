# Mihir Singh and Jake Gilbert
# Creates AI models on dataset

#imports
import dataProcessing
import numpy as np
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# TODO: need to add count to test data from the other kaggle dataset

class Models:
    def __init__(self, trainData, testData):
        # load in file names
        self.trainData = trainData
        self.testData = testData
        
        self.dataProcessor = dataProcessing.dataProcessing()

        # clean data for training file
        self.dfTrain = self.dataProcessor.importData(self.trainData)
        self.dataProcessor.fileInfoForTree(self.dfTrain)
        self.dfTrain = self.dataProcessor.cleanData(self.dfTrain, 1, True)
        
        # clean data for testing file
        self.dfTest = self.dataProcessor.importData(testData)
        self.dfTest = self.dataProcessor.cleanData(self.dfTest, 1, False)
        
    def formatData(self):
        # separate training dataframe into target and features lists for training
        targetTrain = self.dfTrain["count"]    
        featuresTrain = self.dfTrain.drop("count", axis=1)
        featuresTrain= featuresTrain.drop("datetime", axis=1)
        featuresTrain= featuresTrain.drop("casual", axis=1)
        featuresTrain= featuresTrain.drop("registered", axis=1)
        
        # separate testing dataframe into target and features lists for testing
        targetTest = self.dfTest["count"]
        featuresTest = self.dfTest.drop("count", axis=1)
        featuresTest = featuresTest.drop("datetime", axis=1)
        
        listOfData = []
        listOfData.append(targetTrain)
        listOfData.append(featuresTrain)
        listOfData.append(targetTest)
        listOfData.append(featuresTest)

        return listOfData
        
        
    def decisionTree(self):
        # separate training dataframe into target and features lists for training
        targetTrain = self.dfTrain["count"]
        featuresTrain = self.dfTrain.drop("count", axis=1)
        
        # create and train tree
        classifierTree = tree.DecisionTreeClassifier()
        classifierTree = classifierTree.fit(featuresTrain, targetTrain)
        
        # separate testing dataframe into target and features lists for testing
        targetTest = self.dfTest["cnt"]
        featuresTest = self.dfTest.drop("cnt", axis=1)
                
        # plot and predict
        tree.plot_tree(classifierTree)
        prediction = classifierTree.predict(featuresTest)
        
    def randomForest(self):
        # separate training dataframe into target and features lists for training
        targetTrain = self.dfTrain["count"]
        featuresTrain = self.dfTrain.drop("count", axis=1)
        
        # create and train random forest classifier
        classifierRF = RandomForestClassifier(random_state=30)
        classifierRF = classifierRF.fit(targetTrain, featuresTrain)
        
        # separate testing dataframe into target and features lists for testing
        targetTest = self.dfTest["cnt"]
        featuresTest = self.dfTest.drop("cnt", axis=1)
        
        # predict
        prediction = classifierRF.predict(featuresTest)
        
    def naiveBayes(self):
        # separate training dataframe into target and features lists for training
        targetTrain = self.dfTrain["count"]
        
        featuresTrain = self.dfTrain.drop("count", axis=1)
        featuresTrain= featuresTrain.drop("datetime", axis=1)
        featuresTrain= featuresTrain.drop("casual", axis=1)
        featuresTrain= featuresTrain.drop("registered", axis=1)

        # create and train naive bayes classifier
        classifierNB = GaussianNB()
        classifierNB = classifierNB.fit(featuresTrain, targetTrain)
        
        # separate testing dataframe into target and features lists for testing
        targetTest = self.dfTest["count"]
        
        featuresTest = self.dfTest.drop("count", axis=1)
        featuresTest = featuresTest.drop("datetime", axis=1)
        
        # printing for debugging
        print(featuresTrain)
        print(featuresTest)
        print(targetTrain)

        # predict
        prediction = classifierNB.predict(featuresTest)
        return prediction
    
model = Models("train.csv","test.csv")    
print("model ")
myarray = model.naiveBayes()
print(myarray[1000:1500])
        

        
        
        
        
        
