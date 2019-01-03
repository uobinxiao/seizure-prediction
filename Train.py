import numpy
numpy.random.seed(1337) # for reproducibility
import tensorflow
tensorflow.random.set_random_seed(1337)
from Feature import *
from Model import *
from Setting import *
import glob
import os
from pandas import DataFrame

class Trainer:
    def __init__(self):
        pass

    def loadData(self, setting_path, name, is_train, overlap_flag = False):
        setting = Setting(path = setting_path, name = name)
        feature = Feature(setting)
        x_fft, y_fft = feature.loadFromDisk(feature_name = "fft", is_train = is_train)
        x_pca, y_pca = feature.loadFromDisk(feature_name = "pca", is_train = is_train)
        if is_train and overlap_flag:
            x_fft, y_fft = feature.overlap(x_fft, y_fft)
            x_pca, y_pca = feature.overlap(x_pca, y_pca)
        x_fft, _ = feature.scaleAcrossTime(x_fft)
        x_pca, _ = feature.scaleAcrossTime(x_pca)

        return x_fft, y_fft, x_pca, y_pca

    def train(self, setting_path, x_fft, y_fft, x_pca, y_pca):
        assert numpy.array_equal(y_fft, y_pca)
        name = setting_path.split(os.path.sep)[-1][:-5]
        setting = Setting(path = setting_path, name = name)
        channels = x_fft.shape[1]
        fft_bins = x_fft.shape[2]
        pca_bins = x_pca.shape[2]
        steps = x_fft.shape[3]
        model = MultiViewModel.getModel(setting, channels = channels, fft_bins = fft_bins, pca_bins = pca_bins, steps = steps)
        x_fft = x_fft.reshape(x_fft.shape[0], x_fft.shape[1] * x_fft.shape[2], x_fft.shape[3], 1)
        x_pca = x_pca.reshape(x_pca.shape[0], x_pca.shape[1] * x_pca.shape[2], x_pca.shape[3], 1)
        y = np_utils.to_categorical(y_fft, 2)
        model.fit({"input1":x_pca, "input2":x_fft}, {"output":y}, epochs = setting.nb_epoch, verbose = 1, batch_size = setting.batch_size, shuffle = True)
        model.save_weights("weights/contest1/" + str(setting.name) + ".h5")

    def predict(self, setting_path, x_fft, y_fft, x_pca, y_pca, contest):
        assert numpy.array_equal(y_fft, y_pca)
        name = setting_path.split(os.path.sep)[-1][:-5]
        setting = Setting(path = setting_path, name = name)
        channels = x_fft.shape[1]
        fft_bins = x_fft.shape[2]
        pca_bins = x_pca.shape[2]
        steps = x_fft.shape[3]
        model = MultiViewModel.getModel(setting, channels = channels, fft_bins = fft_bins, pca_bins = pca_bins, steps = steps)
        x_fft = x_fft.reshape(x_fft.shape[0], x_fft.shape[1] * x_fft.shape[2], x_fft.shape[3], 1)
        x_pca = x_pca.reshape(x_pca.shape[0], x_pca.shape[1] * x_pca.shape[2], x_pca.shape[3], 1)

        if contest == 1:
            contest = "contest1"
        if contest == 2:
            contest = "contest2"
        weight_path = os.path.join("weights", contest, name+".h5")
        print(weight_path)
        model.load_weights(weight_path)
        predictions = model.predict({'input1':x_pca, "input2":x_fft})
        output = predictions[:, 1]
        output = output.tolist()
        y_fft = y_fft.tolist()
        if not isinstance(y_fft[0],str):
            y_fft = [y.decode() for y in y_fft]
        ans = zip(y_fft, output)
        data_frame = DataFrame(data=list(ans), columns=["clip", "preictal"])
        data_frame.to_csv(setting.savePath + name + ".csv", index=False, header = True)

    def predict_contest1(self):
        config_path = "config/contest1/"
        config_path = os.path.join(config_path, "*.json")
        config_list = glob.glob(config_path)
        config_list = sorted(config_list)
        for config in config_list:
            print(config)
            name = config.split(os.path.sep)[-1][:-5]
            x_fft, y_fft, x_pca, y_pca = self.loadData(setting_path = config, name = name, is_train = False)
            self.predict(setting_path = config, x_fft = x_fft, y_fft = y_fft, x_pca = x_pca, y_pca = y_pca, contest = 1)

if __name__ == "__main__":

    trainer = Trainer()
    config_path = "config/contest1/"
    config_path = os.path.join(config_path, "*.json")
    config_list = glob.glob(config_path)
    config_list = sorted(config_list)
    for config in config_list:
        #if "Patient_2" not in config:
        #    continue
        name = config.split(os.path.sep)[-1][:-5]
        is_train = True
        if "Test" in config:
            is_train = False
        x_fft, y_fft, x_pca, y_pca = trainer.loadData(setting_path = config, name = name, is_train = is_train, overlap_flag = True)
        if is_train:
            #trainer.train(config, x_fft, y_fft, x_pca, y_pca)
            x_fft, y_fft, x_pca, y_pca = trainer.loadData(setting_path = config, name = name, is_train = False)
            trainer.predict(setting_path = config, x_fft = x_fft, y_fft = y_fft, x_pca = x_pca, y_pca = y_pca, contest = 1)
        #exit()
