import pandas
import random
import numpy
import glob
import itertools
from scipy import signal
from MatFile import *
import os
import sklearn.preprocessing
import sklearn.decomposition

class Feature:
    def __init__(self, setting):
        self.setting = setting
        self.result_x = None
        self.result_y = None

    def groupIntoBands(self, fft_data, fft_frequency, band_num):
        bands = None
        if band_num == 5:
            bands = [0.5, 4, 8, 15, 30, 128]
        if band_num == 8:
            bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
        frequency_bands = numpy.digitize(fft_frequency, bands)
        if fft_data.ndim > 1:
            channels = fft_data.shape[0]
            result = []
            for i in xrange(channels):
                data_frame = pandas.DataFrame({"fft":fft_data[i], "band":frequence_bands})
                data_frame = data_frame.groupby("band").mean()
                result.append(data_frame.fft[1: -1])

            return result

        data_frame = pandas.DataFrame({"fft":fft_data, "band": frequency_bands})
        data_frame = data_frame.groupby("band").mean()

        return data_frame.fft[1: -1]

    def fft(self, data_x, data_y, band_num = 8, sampling_rate = 400, window_length = 30, stride = 30):
        #data_x's shape is matFileNumber * channels * matdata
        channels = data_x.shape[1]
        data_length = data_x.shape[2] / sampling_rate
        steps = (data_length - window_length) / stride + 1
        new_array = numpy.zeros((data_x.shape[0], channels, int(band_num + 1), int(steps)))
        size = data_x.shape[0]
        for i in range(size):
            for j in range(channels):
                for frame_index, window_index in enumerate(range(0, int(data_length - window_length + 1), stride)):
                    data = data_x[i, j, window_index * sampling_rate:(window_index + window_length) * sampling_rate]
                    fft_data = numpy.log10(numpy.absolute(numpy.fft.rfft(data)))
                    fft_frequency = numpy.fft.rfftfreq(n = data.shape[-1], d = 1.0 / sampling_rate)
                    new_array[i, j, :band_num, frame_index] = self.groupIntoBands(fft_data, fft_frequency, band_num = 8)
                    new_array[i, j, -1, frame_index] = numpy.std(data)
        self.result_x = new_array
        self.result_y = data_y

        return new_array, data_y

    def saveToDisk(self, feature_name, name, is_train):
        assert self.result_x is None or self.result_y is None

        save_path = self.setting.processedDataPath + feature_name
        if is_train:
            os.makedirs(os.path.join(save_path, self.setting.name), exist_ok=True)
            numpy.save(os.path.join(save_path, self.setting.name, str(name) + "_trainX"), self.result_x)
            numpy.save(os.path.join(save_path, self.setting.name, str(name) + "_trainY"), self.result_y)
        else:
            os.makedirs(os.path.join(save_path, self.setting.name), exist_ok=True)
            numpy.save(os.path.join(save_path, self.setting.name, str(name) + "_testX"), self.result_x)
            numpy.save(os.path.join(save_path, self.setting.name, str(name) + "_testY"), self.result_y)

    def loadFromDisk(self, feature_name, is_train):
        save_path = self.setting.processedDataPath + feature_name
        files = None
        if is_train:
            files = glob.glob(os.path.join(save_path, self.setting.name, "*trainX.npy"))
        else:
            files = glob.glob(os.path.join(save_path, self.setting.name, "*testX.npy"))
        files = sorted(files)
        trainX = None
        trainY = None
        testX = None
        testY = None
        if is_train:
            trainX = numpy.load(files[0])
            file_name = files[0].replace("trainX", "trainY")
            trainY = numpy.load(file_name)
        else:
            testX = numpy.load(files[0])
            file_name = files[0].replace("testX", "testY")
            testY = numpy.load(file_name)
        for f in files[1:]:
            if is_train:
                tmp = numpy.load(f)
                trainX = numpy.concatenate((trainX, numpy.load(f)), axis = 0)
                f = f.replace("trainX", "trainY")
                trainY = numpy.concatenate((trainY, numpy.load(f)), axis = 0)
            else:
                testX = numpy.concatenate((testX, numpy.load(f)), axis = 0)
                f = f.replace("testX", "testY")
                testY = numpy.concatenate((testY, numpy.load(f)), axis = 0)
        if is_train:
            self.result_x = trainX
            self.result_y = trainY
            return trainX, trainY
        else:
            self.result_x = testX
            self.result_y = testY
            return testX, testY

    def overlap(self, x, y):
        shape_x = x.shape
        shape_y = y.shape
        zero_indics = numpy.where(y == 0)
        one_indics = numpy.where(y == 1)
        zero_indics = numpy.array(zero_indics[0]).tolist()
        one_indics = numpy.array(one_indics[0]).tolist()
        first_part = int(shape_x[3] / 2)
        numpy.newaxis
        tmp_x = numpy.concatenate((x[zero_indics[0], :, :, :first_part], x[zero_indics[1], :, :, first_part:]), axis = 2)
        tmp_x = tmp_x.reshape(1, tmp_x.shape[0], tmp_x.shape[1], tmp_x.shape[2])
        tmp_y = []
        tmp_y.append(0)
        for i in range(1, len(zero_indics) - 1):
            if (i + 1) % 6 != 0:
                tmp = numpy.concatenate((x[zero_indics[i], :, :, :first_part], x[zero_indics[i + 1], :, :, first_part:]), axis = 2)
                tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
                tmp_x = numpy.concatenate((tmp_x, tmp), axis = 0)
                tmp_y.append(0)

        for i in range(len(one_indics) - 1):
            if(i + 1) % 6 != 0:
                tmp = numpy.concatenate((x[one_indics[i], :, :, :first_part], x[one_indics[i + 1], :, :, first_part:]), axis = 2)
                tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1], tmp.shape[2])
                tmp_x = numpy.concatenate((tmp_x, tmp), axis = 0)
                tmp_y.append(1)
        x = numpy.concatenate((x, tmp_x), axis = 0)
        y = numpy.concatenate((y, numpy.asarray(tmp_y)), axis = 0)

        return x, y

    def pca(self, data_x, data_y, band_num = 8, sampling_rate = 400, window_length = 30, stride = 30):
        #data_x's shape is mat_file_number * channels * mat_data
        channels = data_x.shape[1]
        data_length = data_x.shape[2] / sampling_rate
        steps = (data_length - window_length) / stride + 1
        new_array = numpy.zeros((data_x.shape[0], channels, channels, int(steps)))
        pca = sklearn.decomposition.PCA()
        size = data_x.shape[0]

        for i in range(size):
            for frame_index, window_index, in enumerate(range(0, int(data_length - window_length + 1), stride)):
                data = data_x[i, :, window_index * sampling_rate:(window_index + window_length) * sampling_rate]
                pca_data = pca.fit_transform(data)
                pca_data = numpy.log10(numpy.absolute(pca_data))
                new_array[i, :, :, frame_index] = pca_data
        self.result_x = new_array
        self.result_y = data_y

        return self.result_x, self.result_y

    def shuffle(self,dataX1, dataX2, dataY):
        resultArray = zip(dataX1, dataX2, dataY)
        random.shuffle(resultArray)
        tempX1, tempX2, tempY = zip(*resultArray)
        tempX1 = numpy.array(tempX1, dtype="float32")
        tempX2 = numpy.array(tempX2, dtype="float32")
        tempY = numpy.array(tempY, dtype="int8")

        return tempX1, tempX2, tempY

    def scaleAcrossTime(self, data, scalers = None):
        sample_num = data.shape[0]
        channel_num = data.shape[1]
        bin_num = data.shape[2]
        time_step_num = data.shape[3]

        if scalers is None:
            scalers = [None] * channel_num
        for i in range(channel_num):
            dataI = numpy.transpose(data[:, i, :, :], axes = (0, 2, 1))
            dataI = dataI.reshape((sample_num * time_step_num, bin_num))
            if scalers[i] is None:
                scalers[i] = sklearn.preprocessing.StandardScaler()
                scalers[i].fit(dataI)
            dataI = scalers[i].transform(dataI)
            dataI = dataI.reshape((sample_num, time_step_num, bin_num))
            dataI = numpy.transpose(dataI, axes = (0, 2, 1))
            data[:, i, :, :] = dataI

        return data, scalers

if __name__ == "__main__":
    pass
