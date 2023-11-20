import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def show_image(imagen, title):
   plt.figure()
   plt.suptitle(title)
   plt.imshow(imagen, cmap = "Greys")
   plt.show()

def plot_X(X, title, fila, columna):
  plt.title(title)
  plt.plot(X)
  plt.xscale(xscale)
  plt.yscale(yscale)
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

features_fila_col = X_train[:, fila, columna]
print(len(np.unique(features_fila_col)))
title = "Valores en (" + str(fila) + ", " + str(columna) + ")"
plot_X(features, title, fila, columna)
