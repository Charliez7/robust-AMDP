import os.path
import random

from tqdm import trange
import matplotlib.pyplot as plt
from functions import *
import tqdm
import xlsxwriter
import numpy as np


# generating ambiguity size for each sa-pairs
def randkappa(n, m):
    kappa = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            kappa[i, j] = round(random.uniform(0.0, 0.3), 2)
    return kappa


# loading data
transition_matrix = np.load('transition_matrix.npy')
branch_matrix = np.load('branch_matrix.npy')
cost = np.load('cost.npy')
p_init = np.load('p_init.npy')

# generating ambiguity set and initial policy
s_num, a_num, _ = transition_matrix.shape
pi = initial_policy(s_num, a_num)
kappa = randkappa(s_num, a_num)

# baseline computation
robust_vi = rvi_sa(transition_matrix, kappa, cost, p_init, branch_matrix)

# parameter initialization
threshold = 0.5
beta = 1
error_j = []
error_j.append(1)
iteration_number=300
for i in trange(iteration_number):
    pi_now, v_now, another_j = outer_pgd(kappa, pi, transition_matrix, cost, branch_matrix, beta, threshold, 'sa')
    pi = pi_now
    threshold *= 0.5
    beta *= 0.95
    error_j.append(abs(another_j - robust_vi) / robust_vi)
    print(error_j[-1])

# saving results
workbook = xlsxwriter.Workbook("error.xlsx")
worksheet = workbook.add_worksheet()
for row_num, row_data in enumerate(error_j):
    for col_num, value in enumerate(row_data):
        worksheet.write(row_num, col_num, value)

workbook.close()
