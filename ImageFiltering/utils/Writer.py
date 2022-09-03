import csv
import numpy as np
class Writer:
    def __init__(self):
        pass
    def newDataCsv(self , X_samp_new , y_samp_new , header):
        self.X_samp_new = X_samp_new
        self.y_samp_new = y_samp_new
        self.header = header
        y_resize = y_samp_new.copy()
        y_resize.shape = (len(y_resize), 1)
        new_samples = np.concatenate((X_samp_new, y_resize), axis=1)

        np.savetxt("new_data.csv", new_samples, fmt="%.5f", delimiter=",", header=",".join(header))


