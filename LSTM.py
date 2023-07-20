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

   def brute_force_knapsack(self, weights, prices, capacity):
    item_count = weights.shape[0]
    picks_space = 2 ** item_count
    best_price = -1
    best_picks = np.zeros(item_count)
    for p in range(picks_space):
        picks = [int(c) for c in f"{p:0{item_count}b}"]
        price = np.dot(prices, picks)
        weight = np.dot(weights, picks)
        if weight <= capacity and price > best_price:
            best_price = price
            best_picks = picks
    return best_picks

   def create_knapsack(self, item_count=5):
     weights = np.random.randint(1, 45, item_count)
     prices = np.random.randint(1, 99, item_count)
     capacity = np.random.randint(1, 99)
     y = self.brute_force_knapsack(weights, prices, capacity)
     return weights, prices, capacity, y


   def norm(self):
     weights, prices, capacity, y = self.create_knapsack()
     weights = weights / capacity
     prices = prices / max(prices)

     otnosh = [weights[i] / prices[i] for i in range (len(prices))]
     return weights, prices, y, otnosh



   def create_knapsack_dataset(self, count, item_count = 5):
     x1 = []
     xx1 = []
     x2 = []
     xx2 = []
     y = []
     yy = []
     x3 = []
     xx3 = []
     fulness = []
     value = []
     ful = []
     val = []

     for i in range(count):
       w, p, e, otn = self.norm()
       for ttt in range(item_count):
         z = 0
         oo = 0
         y_tmp  = e
         y1_tmp = []
         for ii in range(item_count):
           if ii <= ttt:
             y1_tmp.append(1)
           if ii > ttt:
             y1_tmp.append(0)
         y_tmp = [y_tmp[iiii] * y1_tmp[iiii] for iiii in range(item_count)]
         yy.append(y_tmp)
         for ppp in range(item_count):
           z = z + w[ppp] * y1_tmp[ppp]
           oo = oo + p[ppp] * y1_tmp[ppp]
         fulness.append(z)
         value.append(oo)
         xx1.append(w)
         xx2.append(p)
         xx3.append(otn)



       y.append(yy)
       ful.append(fulness)
       val.append(value)
       x1.append(xx1)
       x2.append(xx2)
       x3.append(xx3)
       yy = []
       fulness = []
       value = []
       xx1 = []
       xx2 = []

     return x1,x2,x3,ful,val, y



class Network:
  def __init__(self, inputs_weights, inputs_prices, inputs_otnosh,inputs_fulness, inputs_value, items):
    self.inputs_weights = inputs_weights
    self.inputs_prices = inputs_prices
    self.inputs_otnosh = inputs_otnosh
    self.inputs_fulness = inputs_fulness
    self.inputs_value = inputs_value
    self.items = items

  def mmodel(self,item_data = 5):
    inputs_weights = tf.keras.layers.Input(shape = (item_data, item_data,))
    inputs_prices = tf.keras.layers.Input(shape = (item_data, item_data),)
    inputs_otnosh = tf.keras.layers.Input(shape = (item_data, item_data),)
    inputs_fulness = tf.keras.layers.Input(shape = (item_data, item_data),)
    inputs_value = tf.keras.layers.Input(shape = (item_data, item_data),)
    inputs = Concatenate()([inputs_weights, inputs_prices,inputs_otnosh, inputs_fulness, inputs_value])
    its = LSTM(units = 30, input_shape= (item_data, 25), return_sequences=True)(inputs)
    its =  Dense(20, activation = "relu")(its)
    its =  Dense(10, activation = "relu")(its)
    its =  Dense(item_data, activation="sigmoid")(its)
    model = Model(inputs=[inputs_weights, inputs_prices, inputs_otnosh, inputs_fulness, inputs_value], outputs=[its])
    return model

  def comp(self, x, y, epochs):
    model = self.mmodel()
    z = model.compile(optimizer=tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return z


Firstt = Subjects()



train_x, train_z,train_otn,train_fulness,train_value, train_y = Firstt.create_knapsack_dataset(200)
test_x,test_z,test_otn,test_fulness,test_value, test_y = Firstt.create_knapsack_dataset(5)


TTT = Network(train_x,train_z,train_otn,train_fulness,train_value, train_y)

epochs = 8

train_x = np.array(train_x,dtype='float64')
train_y = np.array(train_y,dtype='float64')
train_z = np.array(train_z,dtype='float64')
train_otn = np.array(train_z,dtype='float64')
train_fulness = np.array(train_y,dtype='float64')
train_value = np.array(train_z,dtype='float64')
test_x = np.array(test_x,dtype='float64')
test_y = np.array(test_y,dtype='float64')
test_z = np.array(test_z,dtype='float64')
test_otn = np.array(test_z,dtype='float64')
test_fulness = np.array(test_y,dtype='float64')
test_value = np.array(test_z,dtype='float64')
print(np.shape(train_x))
print(np.shape(train_z))
print(np.shape(train_otn))
print(np.shape(train_fulness))
print(np.shape(train_value))
print(np.shape(train_y))


M = TTT.mmodel()

M.compile(optimizer=tf.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

M.fit([train_x,train_z, train_otn, train_fulness,train_value], train_y, epochs=10)

print(M.summary())
M.evaluate([test_x,test_z, test_otn, test_fulness,test_value], test_y, batch_size=128)
