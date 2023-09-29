###################################
# TASK 1 Numpy #
###################################

#a) Import NumPy in a python script.
import numpy as np

#b) Think about which values are in the NumPy array ‘d’, then verify if you were correct.
a = np.full((2, 3), 4)
print('Printing a:')
print(a)
b = np.array([[1, 2, 3], [4, 5, 6]])
print('Printing b:') 
print(b)
c = np.eye(2, 3) 
print('Printing c:')
print(c)
d = a + b + c
print('Printing d:')
print(d)

#c) Sum the rows of ‘a’
a = np.array([[1,2,3,4,5],              [5,4,3,2,1],               [6,7,8,9,0],               [0,9,8,7,6]])
print('Summing the rows of a:')
print(a.sum(axis=0))

#d) Get the transpose of a
print('Getting the transpose of a:')
print(a.transpose(1,0))
#print(np.transpose(a))


###################################
# TASK 2 pandas #
###################################
#a) Import pandas.
import pandas as pd

#b) Read the file ‘auto.csv’.
file_path = "C:/Users/Steffen/DeepLearning/Exercise0/auto.csv"
auto_data = pd.read_csv(file_path)

#c) Remove all rows with ‘mpg’ lower than 16
# Filter the rows based on the 'mpg' column
auto_data_filtered_mpg = auto_data[auto_data['mpg'] >= 16]
auto_data.to_csv(file_path, index=False)

#D) Get the first 7 rows of the columns’ weights’ and ‘acceleration’.
rows7 = auto_data_filtered_mpg[['weight', 'acceleration']].head(7)
print(rows7)

#E) Remove the rows in the ‘horsepower’ column that has the value ‘?’, and convert the column to an ‘int’ type instead of a ‘string’.
auto_data = auto_data[auto_data['horsepower'] != '?']
auto_data['horsepower'] = auto_data['horsepower'].astype(int)

auto_data.to_csv(file_path, index=False)

#F) Calculate the averages of every column, except for ‘name’.
averages_except_name = auto_data.drop(columns=['name']).mean()
print(averages_except_name)


###################################
# TASK 3 matplotlib #
###################################

# A) Import Matplotlib
import matplotlib.pyplot as plt

# B) Make a plot with two lines, using ‘a’ and ‘b’. Name the first axis ‘epochs’ and the 2nd axis ‘accuracy’. Call the line made from ‘a’ for ‘training
# accuracy’ and the line made from ‘b’ for ‘validation accuracy’. Title the plot ‘Training and validation accuracy’ and show the plot.
a = np.array([1,1,2,3,5,8,13,21,34])
b = np.array([1,8,28,56,70,56,28,8,1])

plt.plot(a, label = 'training accuracy')
plt.plot(b, label = 'validation acurracy')
plt.xlabel('epochs')
plt.ylabel('acurracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

###################################
# TASK 3 PyTorch #
###################################

#A) Import Pytorch
import torch

#B) Create two random matrices using PyTorch’s (torch.rand)  of size (3x3).
matrix1 = torch.rand(3, 3)
matrix2 = torch.rand(3, 3)
print(matrix1)
print(matrix1)

#C) Multiply the two matrices using PyTorch's matrix multiplication function (torch.matmul)
result = torch.matmul(matrix1,matrix2)
print(result)