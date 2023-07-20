!pip install spektral

import spektral
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from spektral.data import Dataset, DisjointLoader, Graph
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping
from spektral.data.loaders import SingleLoader
from keras import metrics
from keras import optimizers
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.regularizers import l2
from spektral.layers import ChebConv
from keras.layers import Flatten


class BinarySearchTreeNode:
    def __init__(self, data, weight, density, labels, serial_number):
        self.data = data
        self.weight = weight
        self.density = density
        self.labels = labels
        self.serial_number = serial_number
        self.left = None
        self.right = None



    def add_child(self, data, weight, density, labels, matrix, i):
        if data == self.data and weight == self.weight and density == self.density:
            return # node already exist

        if  density < self.density:
            if self.left:
                self.left.add_child(data,weight, density,labels, matrix, i)
            else:
                self.left = BinarySearchTreeNode(data,weight, density, labels, i)
                matrix[i][self.serial_number] = 1
                matrix[self.serial_number][i] = 1

        else:
            if self.right:
                self.right.add_child(data,weight, density, labels, matrix, i)
            else:
                self.right = BinarySearchTreeNode(data,weight, density, labels, i)
                matrix[i][self.serial_number] = 1
                matrix[self.serial_number][i] = 1



    def search(self, val,weight, dens):
        if self.data == val and self.weight == weight and self.density == dens:
            return True

        if dens < self.density:
            if self.left:
                return self.left.search(val,weight, dens)
            else:
                return False

        if dens > self.density:
            if self.right:
                return self.right.search(val,weight, dens)
            else:
                return False

    def in_order_traversal(self):
        elements = []
        if self.left:
            elements += self.left.in_order_traversal()

        elements.append([self.data, self.weight,self.density])

        if self.right:
            elements += self.right.in_order_traversal()

        return elements


def build_tree(elements, matrix):
    #print("Building tree with these elements:",elements)
    root = BinarySearchTreeNode(elements[0][0], elements[1][0], elements[2][0], elements[3][0], 0)

    for i in range(1,len(elements[0])):
        root.add_child(elements[0][i], elements[1][i], elements[2][i], elements[3][i], matrix, i)

    return root


class MyDataset(Dataset):

    def __init__(self, n_samples, item_count = 4000, **kwargs):
        self.n_samples = n_samples
        self.item_count = item_count
        super().__init__(**kwargs)


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

    def create_knapsack(self):
      weights = np.random.randint(10, 80, self.item_count)
      prices = np.random.randint(1, 599, self.item_count)
      capacity =np.random.randint(62000, 63990)
      y = self.knapsack(weights, prices, capacity)
      return weights, prices, capacity, y


    def norm(self):
      weights, prices, capacity, y = self.create_knapsack()

      weights = weights / capacity
      prices = prices / max(prices)

      otnosh = [weights[i] / prices[i] for i in range (len(prices))]

      y = np.array(y, dtype = "float32")

      return weights, prices, y, otnosh


    def read(self):
        def make_graph():

            a = np.zeros((self.item_count,self.item_count))

            w, p, yy, otn = self.norm()

            numbers_tree = build_tree([p,w,otn, yy], a)

            # Node features
            x = np.zeros((3, self.item_count))
            x[0][:] = w
            x[1][:] = p
            x[2][:] = otn
            x = x.transpose()

            # Edges
            a = sp.csr_matrix(a)

            # Labels
            y = np.zeros((self.item_count, 1))
            for i in range(self.item_count):
              y[i][0] = yy[i]


            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph() for _ in range(self.n_samples)]




data = MyDataset(1)

train_mask, val_mask, test_mask = np.zeros(data.n_nodes), np.zeros(data.n_nodes),np.zeros(data.n_nodes)

for i in range(data.n_nodes):
  if i < data1.n_nodes * 0.4:
    train_mask[i] = 1
    val_mask[i] = 0
    test_mask[i] = 0
  if i < data.n_nodes * 0.7 and i >= data.n_nodes * 0.4:
    train_mask[i] = 0
    val_mask[i] = 1
    test_mask[i] = 0
  if i >= data.n_nodes * 0.7:
    train_mask[i] = 0
    val_mask[i] = 0
    test_mask[i] = 1


def mask_to_weights(mask):
    return mask / np.count_nonzero(mask)

weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (train_mask, val_mask, test_mask)

)

channels = 1000  # Number of channels in the first layer
K = 2  # Max degree of the Chebyshev polynomials
dropout = 0.49  # Dropout rate for the features
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 1e-2  # Learning rate
epochs = 150  # Number of training epochs
patience = 20  # Patience for early stopping
a_dtype = data1[0].a.dtype  # Only needed for TF 2.1

N = data1.n_nodes  # Number of nodes in the graph
F = data1.n_node_features  # Original size of node features
n_out = 1 # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)



#do_1 = Dropout(dropout)(x_in)
gc_1 = ChebConv(
    channels, K=K, activation="relu", kernel_regularizer=l2(l2_reg), use_bias=False
)([x_in, a_in])

do_2 = Dropout(dropout)(gc_1)
gc_2 = ChebConv(500, K=K, activation="relu", use_bias=False)([do_2, a_in])

do_3 = Dropout(dropout)(gc_2)
gc_3 = ChebConv(250, K=K, activation="relu", kernel_regularizer=l2(l2_reg), use_bias=False)([do_3, a_in])
do_4 = Dropout(dropout)(gc_3)
gc_4 = ChebConv(100, K=K, activation="relu", use_bias=False)([do_4, a_in])

# Output
output1 = Flatten()(gc_4)
ou3 = Dense(600, activation="relu")(output1)
ou4 = Dense(200, activation="relu")(ou3)
ou = Dense(20, activation="relu")(ou3)
ou21 = Dense(8, activation="relu")(ou)
output = Dense(n_out, activation="sigmoid")(ou21)





# Build model
model = Model(inputs=[x_in, a_in], outputs=output)
optimizer = Adam()
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=BinaryCrossentropy(),
    weighted_metrics=[metrics.binary_accuracy],
)
model.summary()

loader_tr = SingleLoader(data, sample_weights=weights_tr)
loader_va = SingleLoader(data, sample_weights=weights_va)

history = model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],

)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(data, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
