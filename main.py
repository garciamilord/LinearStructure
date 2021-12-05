# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from array import array

import numpy as np


def norm():
    x=3
    y=1
    z=2
    norm = math.sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    return print('Question 1: Part 1: Compute the norm of (3,1,2): \n'+ str(norm))

def transposeProduct():
    Amatrix = np.array([[4, -2, 1], [5, 7, -1]])
    Tmatrix =Amatrix.T
    Xvector = np.array([[2],[3]])
    Product = Tmatrix @ Xvector
    return print('Question 2: Part 1: Compute the product AT x given that: \n'+ str(Product))

def determinant():
    matrix = np.array([[3,4,3],[-1,9,-1],[7,2,7]])
    det = np.linalg.det(matrix)
    return print('Question 3: Part 1: Compute determinant: \n'+str(det))

#Part II

def matrixMultipcation():
    a = np.array([[-1.5,3,2],[1,-1,0]])
    b = np.array([[-1,-1],[0,2],[1,0]])
    c = a @ b
    return print("Question 1: Part 2: numpy matrix multiplication: \n"+str(c))

def myMatrixMultipcation():
    #2x3
    Am = [[-1.5,3,2],[1,-1,0]]
    #3x2
    Bm = [[-1,-1],[0,2],[1,0]]
    print("Question 1: Part 2: my matrix multiplication: ")
    result = [[sum(a * b for a, b in zip(A_row, B_col))
               for B_col in zip(*Bm)]
                    for A_row in Am]
    for r in result:
        print(r)


def systemofequation():
    # 2* x1 + 2* x2 + 6* x3 =24
    # 2* x1 - 2* x2 - 2* x3 =0
    # 4* x1 + 2* x2 - 4* x3 =6
    A = np.array([[2,2,6],[2,-2,-2],[4,2,-4]])
    B = np.array([24,0,6])
    x = np.linalg.solve(A,B )
    return print("Question 2: Solve the following system of equations: \n"+str(x))


if __name__ == '__main__':
    norm()
    print()
    transposeProduct()
    print()
    determinant()
    print()
    matrixMultipcation()
    print()
    myMatrixMultipcation()
    print()
    systemofequation()