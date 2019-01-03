import glob
import numpy
from Setting import *
from MatFile import *
from Subject import *
from SignalUtils import *
class Data:
    def __init__(self):
        self.mat = MatFile()
        self.subject = Subject()
        self.signal = SignalUtils()
        self.subject_name = ""

    def processMat(self, data_file, is_train, contest):
        print(data_file)
        self.mat.readMat(data_file, contest)
        self.mat.name = data_file
        self.mat.sampling_rate = 400
        if contest == 1:
            self.mat.data = self.signal.resample(self.mat.data, 400)
        self.mat.data = self.signal.butterWorthBandpassFilter(self.mat.data, band=[0.1, 180], frequency = 400)
        mat_list = self.mat.getDataList(time_length = self.mat.time_length)
        list_length = mat_list.shape[0]
        sizex, sizey = mat_list[0].data.shape
        matBagX = numpy.zeros((list_length, sizex, sizey))
        matBagY = []

        for i in range(0, list_length):
            matBagX[i, :, :] = mat_list[i].data
            if is_train:
                matBagY.append(self.mat.getMatLabel())
            else:
                matBagY.append(self.mat.name.split("/")[-1])

        return matBagX, matBagY

    def processMatList(self, mat_list, contest, sequence = 0, is_train = True):
        dim0 = len(mat_list)
        assert dim0 != 0
        X, Y = self.processMat(mat_list[0], is_train, contest)
        xdim0, xdim1, xdim2 = X.shape
        dim0 = dim0 * xdim0

        data_x = numpy.zeros((dim0, xdim1, xdim2))
        data_y = []
        data_x[0: xdim0, :, :] = X
        data_y += Y

        for i in range(1, len(mat_list)):
            tmp_x, tmp_y = self.processMat(mat_list[i], is_train, contest)
            data_x[i * xdim0: (i + 1) * xdim0, :, :] = tmp_x
            data_y += tmp_y
        if contest == 2:
            data_x = numpy.transpose(data_x, (0, 2, 1))

        return data_x, data_y

    def processSubject(self, subject, is_train, contest, split_number = 1, sequence = 0):
        data_list = None
        if is_train:
            data_list = self.subject.getTrainFileList(contest = contest, name = subject)
        else:
            data_list = self.subject.getTestFileList(contest = contest, name = subject)
        if contest == 1:
            if subject.find("Dog") != -1:
                index = subject.find("Dog")
                self.subject_name = subject[index: index + len("Dog") + 2]
            else:
                index = subject.find("Patient")
                self.subject_name = subject[index: index + len("Patient") + 2]
        amount = None
        if len(data_list) % split_number == 0:
            amount = int(len(data_list) / split_number)
        else:
            amount = int(len(data_list) / split_number + 1)
        for i in range(split_number):
            if i == sequence:
                return self.processMatList(data_list[i * amount: (i + 1) * amount], contest = contest, sequence = sequence, is_train = is_train)

if __name__ == "__main__":
    data = Data()
    data.processSubject("/home/xiaobin/Downloads/train_1", is_train = True)
