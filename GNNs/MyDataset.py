from GNNs.BinarySearchTreeNode import BinarySearchTreeNode
import spektral
from spektral.data import Dataset
import numpy as np

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