import csv
import numpy as np
class Reader:
    def __init__(self , fileName):
        self.fileName = fileName
    def construct(self):
        file = open(self.fileName)
        csvreader = csv.reader(file)
        header = next(csvreader)
        header = header[:-1]
        rows = []
        for row in csvreader:
            rows.append(row)

        file.close()

        X = []
        y = []
        for row in rows:
            row = row[:-1]
            row = [float(i) for i in row]
            X.append(row[:-1])
            y.append(row[-1])
        X = np.asarray(X)
        y = np.asarray(y)
        return X , y , header
