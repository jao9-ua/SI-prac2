import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def show_image(imagen, title):
   plt.figure()
   plt.suptitle(title)
   plt.imshow(imagen)
   plt.show()

def plot_X(X, title):
  plt.title(title)
  plt.plot(X)
  plt.xscale("linear")
  plt.yscale("linear")
  plt.show()



(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)

for i in range(3):
 title = "Mostrando imagen X_train[" + str(i) + "]"
 title = title + " -- Y_train[" + str(i) + "] = " + str(Y_train[i])
 show_image(X_train[i], title)

fila=5
columna=5

#  Extrae valores de un pixel especifico
features_fila_col = X_train[:, fila, columna]
#Calcula valores unicos que existen en este pixel
print(len(np.unique(features_fila_col)))
title = "Valores en (" + str(fila) + ", " + str(columna) + ")"
plot_X(features_fila_col, title)
cant_seises = np.count_nonzero(Y_train == 6)

print("Cantidad de seises: ", cant_seises)
