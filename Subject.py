from MatFile import *
import os
import glob

class Subject:
    def __init__(self, name=""):
        self.name = name
        self.matTrainFileList = []
        self.matTestFileList = []

    def getTrainFileList(self, contest, name=""):
        self.matTrainFileList = []
        if self.name == "":
            self.name = name
        trainFiles = glob.glob(os.path.join(self.name, "*.mat"))
        if contest == 1:
            trainFiles = sorted(trainFiles)
            for trainFile in trainFiles:
                if "_test_" not in trainFile:
                    self.matTrainFileList.append(trainFile)
        if contest == 2:
            trainFiles = sorted(trainFiles, key = lambda x: int(x.split("/")[-1].split("_")[1]))
            self.matTrainFileList = trainFiles
        return self.matTrainFileList

    def getTestFileList(self, contest, name=""):
        self.matTestFileList = []
        if name[-1] == '/':
            name = name[0: -1]
        if self.name == "":
            self.name = name
        testFiles = glob.glob(self.name + "/*.mat")
        testFiles = sorted(testFiles)
        if contest == 1:
            testFiles = sorted(testFiles)
            for testFile in testFiles:
                if "_test_" in testFile:
                    self.matTestFileList.append(testFile)
        if contest == 2:
            testFiles = sorted(testFiles, key = lambda x: int(x.split("/")[-1].split("_")[1]))
            self.matTestFileList = testFiles

        return self.matTestFileList

if __name__ == "__main__":
    subject = Subject()
    subject.getTrainFileList("/Users/xiaobin/Downloads/train_1/")
