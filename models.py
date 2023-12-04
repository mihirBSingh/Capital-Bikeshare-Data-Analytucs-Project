# Mihir Singh and Jake Gilbert
# Creates AI models on dataset

#imports
import dataProcessing
from sklearn import tree

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
        self.dataProcessorTrain.fileInfoForTree(self.dfTest)
        self.dfTest = self.dataProcessorTrain.cleanData(self.dfTest, 1, False)
        
        
    def decisionTree(self):
        # separate dataframe into target and features lists for training
        targetTrain = self.dfTrain["count"]
        featuresTrain = self.dfTrain.drop("count", axis=1)
        
        # create and train tree
        classifierTree = tree.DecisionTreeClassifier()
        classifierTree = classifierTree.fit(featuresTrain, targetTrain)
        
        # separate dataframe into target and features lists for testing
        targetTest = self.dfTest["count"]
        featuresTest = self.dfTest.drop("count", axis=1)
                
        # plot and predict
        tree.plot_tree(classifierTree)
        prediction = classifierTree.predict(featuresTest)
        
        
        
