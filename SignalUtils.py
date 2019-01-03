import scipy.signal
import sklearn
import numpy
import pywt

class SignalUtils:
    def __init__(self, name=""):
        self.name = name

    def resample(self, data, newSamplingRate, timeLength = 600):

        return scipy.signal.resample(data,newSamplingRate * timeLength, axis = 1)

    def butterWorthBandpassFilter(self,data, order=5, band = [1, 47], frequency = 400):
        b, a  = scipy.signal.butter(order, numpy.array(band) / (frequency/2.0), btype="band")

        return scipy.signal.lfilter(b, a, data, axis=1)

    def fft(self, data):
        axis = data.ndim - 1

        return numpy.fft.rfft(data, axis = axis)

    def log10(self, data):

        index = numpy.where(data < 0)
        data[index] = numpy.max(data)
        data[index] = numpy.min(data) * 0.1

        return numpy.log10(data)
