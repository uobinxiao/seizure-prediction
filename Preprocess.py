from Feature import *
from Setting import *
from Data import *
import glob

def thread_job(path, setting, sequence, is_train, feature_name):
        X_train, y_train = data.processSubject(path, is_train = is_train, contest = contest, split_number = setting.splitNum, sequence = sequence)
        feature = Feature(setting)
        X_pca_train, y_pca_train = feature.pca(X_train, y_train, window_length = 30)
        feature.saveToDisk(feature_name = "pca", name = str(sequence), is_train = is_train)
        X_fft_train, y_fft_train = feature.fft(X_train, y_train, window_length = 30)
        feature.saveToDisk(feature_name = "fft", name = str(sequence), is_train = is_train)

def preprocess(config_path, contest):
    config_path = os.path.join(config_path, "*.json")
    config_list = glob.glob(config_path)
    config_list = sorted(config_list)
    for config in config_list:
        is_train = True
        if "Test" in config:
            is_train = False
        name = config.split(os.path.sep)[-1][:-5]
        setting = Setting(path = config, name = name)
        path = setting.rawDataPath
        path = os.path.join(path, name)
        data = Data()
        feature = Feature(setting)
        for sequence in range(setting.splitNum):
            X_train, y_train = data.processSubject(path, is_train = is_train, contest = contest, split_number = setting.splitNum, sequence = sequence)
            X_pca_train, y_pca_train = feature.pca(X_train, y_train, window_length = 30)
            feature.saveToDisk(feature_name = "pca", name = str(sequence), is_train = is_train)
            X_fft_train, y_fft_train = feature.fft(X_train, y_train, window_length = 30)
            feature.saveToDisk(feature_name = "fft", name = str(sequence), is_train = is_train)

if __name__ == "__main__":
    preprocess(config_path = "config/contest1/", contest = 1)
