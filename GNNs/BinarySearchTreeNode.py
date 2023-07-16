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
