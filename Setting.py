import json
import os
class Setting:
    def __init__(self, path, name):
        #data process related
        self.splitNum = 1
        self.timeSlotLength = 60 # in seconds
        self.timeSlotNum = 600 / self.timeSlotLength
        self.rawDataPath = None
        self.processedDataPath = None
        self.savePath = None
        self.resampleFrequency = None

        #cnn model related
        self.name = name
        self.nb_filter = None
        self.nb_epoch = None
        self.batch_size = None
        self.output1 = None
        self.output2 = None
        self.lr = None
        self.dropout = None
        self.l2 = None

        if path != "":
            self.path = path
            if os.path.isfile(path):
                self.loadSettings(path, name)

    def loadSettings(self, path, name):
        data = None
        with open(self.path, "r") as yfile:
            data = yfile.read().replace('\n', '')

        setting_dict = json.loads(data)
        self.name = setting_dict['name']
        self.resampleFrequency = setting_dict['resampleFrequency']
        self.rawDataPath = setting_dict['rawDataPath']
        self.processedDataPath = setting_dict['processedDataPath']
        self.splitNum = setting_dict['splitNum']
        self.timeSlotLength = setting_dict['timeSlotLength']
        self.timeSlotNum = setting_dict['timeSlotNum']
        self.nb_filter = setting_dict['nb_filter']
        self.nb_epoch = setting_dict['nb_epoch']
        self.batch_size = setting_dict['batch_size']
        self.output1 = setting_dict['output1']
        self.output2 = setting_dict['output2']
        self.lr = setting_dict['lr']
        self.dropout = setting_dict['dropout']
        self.l2 = setting_dict['l2']
        self.savePath = setting_dict['savePath']

        return self

    def saveSettings(self, path = ""):
        data = dict(Setting = dict(name = self.name,
                                   DataProcess = dict(rawDataPath = self.rawDataPath,
                                                      processedDataPath = self.processedDataPath,
                                                      splitNum = self.splitNum,
                                                      timeSlotLength = self.timeSlotLength,
                                                      timeSlotNum = self.timeSlotNum,
                                                      savePath = self.savePath
                                                      ),
                                   Model = dict(nb_filter = self.nb_filter,
                                                nb_epoch = self.nb_epoch,
                                                batch_size = self.batch_size,
                                                output1 = self.output1,
                                                output2 = self.output2,
                                                lr = self.lr,
                                                dropout = self.dropout,
                                                l2 = self.l2)
                                   ))
        if path != "":
            self.path = path
        with open("settings.json", "a") as outfile:
            outfile.write(json.dumps(data, default_flow_style=False))
