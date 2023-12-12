# Mihir Singh and Jake Gilbert
# Creates AI models on dataset

# imported modules
import dataProcessing
import pandas as pd
import numpy as np
import graphviz
import calplot
import matplotlib.pyplot as plt
import csv

# imports for models
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# TODO: need to get plot of decision tree to show up

class Models:
    def __init__(self, trainData, testData):
        # load in file names
        self.trainData = trainData
        self.testData = testData
        
        self.dataProcessor = dataProcessing.dataProcessing()

        # clean data for training file
        self.dfTrain = self.dataProcessor.importData(self.trainData)
        self.dataProcessor.fileInfo(self.dfTrain)
        self.dfTrain = self.dataProcessor.cleanData(self.dfTrain)

        # clean data for testing file
        self.dfTest = self.dataProcessor.importData(testData)
        self.dfTest = self.dataProcessor.cleanData(self.dfTest)

        
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
        # grab formatted dataframes
        dataframes = self.formatData()
        targetTrain = dataframes[0]
        featuresTrain = dataframes[1]
        targetTest = dataframes[2]
        featuresTest = dataframes[3]
        
        # create and train tree
        classifierTree = tree.DecisionTreeClassifier(max_depth= 5, ccp_alpha=0.005) 
        classifierTree = classifierTree.fit(featuresTrain, targetTrain)
               
        # plot
        plotExport = tree.export_graphviz(classifierTree, out_file=None)
        graph = graphviz.Source(plotExport)
        graph.render("decisionTreePlot")
         
        # predict
        prediction = classifierTree.predict(featuresTest)
        return prediction
        
    def randomForest(self):
        # grab formatted dataframes
        dataframes = self.formatData()
        targetTrain = dataframes[0]
        featuresTrain = dataframes[1]
        targetTest = dataframes[2]
        featuresTest = dataframes[3]
        
        # create and train random forest classifier
        classifierRF = RandomForestClassifier(random_state=30)
        classifierRF = classifierRF.fit(featuresTrain, targetTrain)
        
        # predict
        prediction = classifierRF.predict(featuresTest)
        
        # get decision path
        # print("decision path")
        # print(classifierRF.decision_path(featuresTest))
        
        return prediction
        
    def naiveBayes(self):
        # grab formatted dataframes
        dataframes = self.formatData()
        targetTrain = dataframes[0]
        featuresTrain = dataframes[1]
        targetTest = dataframes[2]
        featuresTest = dataframes[3]
        
        # create and train naive bayes classifier
        classifierNB = GaussianNB()
        classifierNB = classifierNB.fit(featuresTrain, targetTrain)
        
        # predict
        prediction = classifierNB.predict(featuresTest)
        
        # get probabilities
        # print("probs")
        # print(classifierNB.predict_proba(featuresTest))
        
        return prediction
    
    def neuralNet(self):
        # grab formatted dataframes
        dataframes = self.formatData()
        targetTrain = dataframes[0]
        featuresTrain = dataframes[1]
        targetTest = dataframes[2]
        featuresTest = dataframes[3]
        
        # create and train neural network
        classifierNN = MLPClassifier(solver='sgd', max_iter=4000)
        classifierNN = classifierNN.fit(featuresTrain, targetTrain)
        
        # predict
        prediction = classifierNN.predict(featuresTest)
        return prediction

    def makeYearPlot(self, prediction):  
        # grab datetimes of test data    
        self.dfTest['datetime'] = pd.to_datetime(self.dfTest["datetime"], format='%Y-%m-%d %H:%M:%S')
        time = self.dfTest['datetime']
        
        # add prediction to frame
        self.dfTest["prediction"] = prediction
        
        # group by day
        result = self.dfTest.groupby(self.dfTest['datetime'].dt.date)['prediction'].mean().reset_index()
        print(result['prediction'][200:250])
        timeResult = pd.to_datetime(result['datetime'], format='%Y-%m-%d %H:%M:%S')
        results = [float(x) for x in result['prediction']]
        
        # create dataframe (really two-column series) for plot
        values = pd.Series(results, index = timeResult)

        # plot
        calplot.calplot(values)
        plt.show()
            
    def evaluate(self):
        # grab formatted dataframes
        dataframes = self.formatData()
        targetTrain = dataframes[0]
        featuresTrain = dataframes[1]
        targetTest = dataframes[2]
        featuresTest = dataframes[3]
        
        # create models
        model = Models("train.csv","test.csv")  
        
        # arrays of data
        DTClassifier = model.decisionTree()
        RFClassifier = model.randomForest()
        NBClassifier = model.naiveBayes()
        NNClassifier = model.neuralNet()
        
        # vars for evaluation
        DTCorrect = 0
        RFCorrect = 0
        NBCorrect = 0
        NNCorrect = 0
        CBCorrect = 0
        length = len(targetTest)
        
        # testing
        # Specify the file path
        csv_file_path = 'output.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Value'])  # Write a header if needed
            csv_writer.writerows(map(lambda x: [x], DTClassifier))
        
        # evaluate
        for i in range(len(targetTest)):
            if(DTClassifier[i] != None or RFClassifier[i] != None or NBClassifier[i] != None or NNClassifier[i] != None):
                if(targetTest[i] == DTClassifier[i]):
                    DTCorrect += 1
                if(targetTest[i] == RFClassifier[i]):
                    RFCorrect += 1
                if(targetTest[i] == NBClassifier[i]):
                    NBCorrect += 1
                if(targetTest[i] == NNClassifier[i]):
                    NNCorrect += 1  
                if(int((DTClassifier[i] + RFClassifier[i] + NBClassifier[i] + NNClassifier[i])/4) == targetTest[i]):
                    CBCorrect +=1
        
        # model accuracies
        DTAccuracy = DTCorrect/length
        RFAccuracy = RFCorrect/length
        NBAccuracy = NBCorrect/length
        NNAccuracy = NNCorrect/length
        CBAccuracy = CBCorrect/length
        
        # bar chart
        accuracies = [DTAccuracy, RFAccuracy, NBAccuracy, NNAccuracy, CBAccuracy]
        print("accuracies: ", accuracies)
        plt.bar(range(len(accuracies)), accuracies)
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracies at Predicting Capital Bikeshare Usage')
        plt.show()
              
# TEST       
# create models
model = Models("train.csv","test.csv")  

# create models
# model.evaluate() # comment out if making plots

# models for making plots
# myarray = model.randomForest() # Random Forest
# myarray = model.decisionTree() # Decision Tree
# myarray = model.naiveBayes()     # Naive Bayes
myarray = model.neuralNet()    # Neural Net

# make year plot of busy-ness
model.makeYearPlot(myarray)


        
        
        
        
        
