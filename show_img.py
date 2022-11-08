import csv
import numpy as np
from matplotlib import pyplot as plt

with open('dataset/mnist_train.csv', newline='') as f:
  reader = csv.reader(f)
  headers = next(reader)
  row1 = [int(x) for x in next(reader)]
  label = row1[0]
  img1 = np.array(row1[1:])
  img1 = np.reshape(img1, (28, 28))
  plt.imshow(img1, interpolation='nearest')
  plt.show()

