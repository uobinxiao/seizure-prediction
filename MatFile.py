import numpy
import scipy.io
import gc

class MatFile:
    def __init__(self, name=""):
        self.patient = ""
        self.name = name
        self.data = None
        self.channels = []
        self.sampling_rate = 400
        self.time_length = 0
        self.sequence_number = 0
        self.mat_type = ""

    def readMat(self, file_name, contest):
        if self.name == "" or file_name != "":
            self.name = file_name
        if contest == 1:
            index = self.name.find("Dog")
            if index == -1:
                index = self.name.find("Patient")
                self.patient = self.name[index:index + len("Patient") + 2]
            else:
                self.patient = self.name[index:index + len("Dog") + 2]
            if "_test_" in file_name:
                self.mat_type = "test"
            elif "preictal" in file_name:
                self.mat_type = "preictal"
            else:
                self.mat_type = "interictal"
            raw_data = scipy.io.loadmat(file_name, squeeze_me = True)
            for key in raw_data.keys():
                if key != "__version__" and key != "__globals__" and key != "__header__":
                    raw_data = raw_data[key]
                    break
            raw_data = raw_data.tolist()
            self.data = numpy.array(raw_data[0], dtype="float32")
            self.time_length = int(raw_data[1])
            self.sampling_rate = int(raw_data[2])
            self.channels = raw_data[3]
            if self.mat_type != "test":
                self.sequence = int(raw_data[4])


        if contest == 2:
            last_name = file_name.split("/")[-1]
            label = last_name.split("_")
            if len(label) == 2:
                self.mat_type = "test"
            else:
                if label[-1] == "0.mat":
                    self.mat_type = "interictal"
                elif label[-1] == "1.mat":
                    self.mat_type = "preictal"
                else:
                    raise("mat type error")
            raw_data = scipy.io.loadmat(file_name, squeeze_me = True)
            for key in raw_data.keys():
                if key != "__version__" and key != "__globals__" and key != "__header__":
                    raw_data = raw_data[key]
                    break
            channels = raw_data.shape[1]
            raw_data = raw_data.tolist()
            self.data = numpy.array(raw_data, dtype="float32")
            self.time_length = 600
            self.sampling_rate = 400
            self.channels = channels
            #kif self.mat_type != "test":
            #k    self.sequence = int(raw_data[4])

    def getMatLabel(self):
        if self.mat_type == "preictal":
            return 1
        elif self.mat_type == "interictal":
            return 0
        else:
            return None

    def getDataList(self, time_length):
        if self.time_length == time_length:
            return numpy.array([self], dtype=numpy.object)
        else:
            mat_list = numpy.ndarray((self.time_length / time_length, ), dtype=numpy.object)
            for i in range(self.time_length / time_length):
                mat = MatFile()
                mat.channels = self.channels
                mat.sampling_rate = self.sampling_rate
                mat.data = self.data[:, time_length * i * self.sampling_rate : time_slot * (i + 1) * self.sampling_rate]
                mat.mat_type = self.mat_type
                mat.name = self.name + "_" + str(i)
                mat.sequence = self.sequence_number
                mat.subject = self.subject
                mat.time_length = self.time_length
                mat_list[i] = mat

            return mat_list
if __name__ == "__main__":
   mat_file = MatFile()
   mat_file.readMat("/Users/xiaobin/Downloads/train_1/1_1000_0.mat")
