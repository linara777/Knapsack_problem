 import numpy as np
 from keras.models import Model
 from keras.layers import Dense, Activation,  Input, Concatenate, Multiply
 from keras.metrics import binary_accuracy
 from keras.losses import binary_crossentropy
 import keras.backend as K
 import tensorflow as tf
 from keras.layers import LSTM
 import time
 from numpy import linalg as LA
 import matplotlib.pyplot as plt
 import matplotlib.lines as mlines
 import matplotlib.lines

 class Subjects:
   def __init__(self):
     pass

   def knapsack(self, weights, prices, maxCapacity):
      n = len(weights)
      dp = [[0 for j in range(maxCapacity+1)] for i in range(n+1)]
      for i in range(1, n+1):
        for j in range(1, maxCapacity+1):
          if weights[i-1] > j:
            dp[i][j] = dp[i-1][j]
          else:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + prices[i-1])
      selected = []
      i, j = n, maxCapacity
      while i > 0 and j > 0:
        if dp[i][j] != dp[i-1][j]:
          selected.append(i-1)
          j -= weights[i-1]
        i -= 1
      selected.reverse()
      y = [0 for i in range(len(weights))]
      for i in selected:
        y[i] = 1
      return y

   def create_knapsack(self, item_count=5):
     weights = np.random.randint(1, 45, item_count)
     prices = np.random.randint(1, 99, item_count)
     capacity = np.random.randint(1, 699)
     y = self.knapsack(weights, prices, capacity)
     return weights, prices, capacity, y


   def norm(self):
     weights, prices, capacity, y = self.create_knapsack()
     weights = weights / capacity
     prices = prices / max(prices)

     otnosh = [weights[i] / prices[i] for i in range (len(prices))]
     return weights, prices, y, otnosh 

   def create_knapsack_dataset(self, count, item_count=5):
    x = []
    y = []
    pr = []
    orn = []
    for i in range(count):
      w, p, e, otn = self.norm()
      x.append(w)
      pr.append(p)
      y.append(e)
      orn.append(otn)
  
    return x,pr,orn,y

  class Network:
  def __init__(self, inputs_weights, inputs_prices, inputs_otnosh, items):
    self.inputs_weights = inputs_weights
    self.inputs_prices = inputs_prices
    self.inputs_otnosh = inputs_otnosh
    self.items = items

  def mmodel(self,item_data = 5):
    inputs_weights = tf.keras.layers.Input((item_data,))
    inputs_prices = tf.keras.layers.Input((item_data),)
    inputs_otnosh = tf.keras.layers.Input((item_data),)
    inputs = Concatenate()([inputs_weights, inputs_prices,inputs_otnosh])
    its =  Dense(35, activation = "relu")(inputs)
    its =  Dense(10, activation = "relu")(its)
    its =  Dense(5, activation="sigmoid")(its)
    model = Model(inputs=[inputs_weights, inputs_prices, inputs_otnosh], outputs=[its])
    return model

  def comp(self, x, y, epochs):
    model = self.mmodel()
    z = model.compile(optimizer=tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return z

Firstt = Subjects()

train_x, train_z, train_otn,train_y = Firstt.create_knapsack_dataset(200)
test_x,test_z,test_otn, test_y, = Firstt.create_knapsack_dataset(50)

TTT = Network(train_x,train_z,train_otn, train_y)

epochs = 8
train_x = np.array(train_x,dtype='float64')
train_y = np.array(train_y,dtype='float64')
train_z = np.array(train_z,dtype='float64')
y_train = np.asarray(train_y).astype('float32').reshape((-1,1))
train_otn = np.array(train_z,dtype='float64')

test_x = np.array(test_x,dtype='float64')
test_y = np.array(test_y,dtype='float64')
test_z = np.array(test_z,dtype='float64')
test_otn = np.array(test_z,dtype='float64')
y_test = np.asarray(test_y).astype('float32').reshape((-1,1))

M = TTT.mmodel()

M.compile(optimizer=tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

M.fit([train_x,train_z, train_otn], train_y, epochs=10)

print(M.summary())
M.evaluate([test_x,test_z, test_otn], test_y, batch_size=128)
