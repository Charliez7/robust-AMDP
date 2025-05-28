import numpy as np
import cvxpy as cp
import xlsxwriter
import os
from tqdm import trange
from functions import *


def randomize_ellipsoid_parameter(s_num, a_num):

    A = np.round(np.random.rand(s_num * a_num * s_num, s_num * a_num * s_num) * 0.1, 2)
    Q = np.dot(A.T, A)

    return Q


transition_matrix = np.load('transition_matrix.npy')
branch_matrix = np.load('branch_matrix.npy')
cost = np.load('cost.npy')
p_init = np.load('p_init.npy')
folder = "nonrec_rob"
s_num, a_num, _ = transition_matrix.shape
pi = initial_policy(s_num, a_num)
pi_1 = pi.copy()

error_j = []
j = []
j.append(1)

non_rob = []
non_rob.append(1)

# Q=randomize_ellipsoid_parameter(s_num, a_num)
Q=np.eye(s_num * a_num * s_num)

threshold=1
theta = 1
iteration_num=100
beta=1
for i in trange(iteration_num):
    pi_now, v_now, another_j = outer_pgd_nonrec(pi, transition_matrix, cost, branch_matrix, beta, threshold, Q, theta)
    pi = pi_now
    threshold*=0.5
    threshold=max(threshold, 1e-4)
    beta*=0.95
    j.append(another_j)
threshold=1
beta=1
for i in trange(iteration_num):
    pi_now, v_now, another_j = outer_pgd_nonrec(pi_1, transition_matrix, cost, branch_matrix, 0.01, threshold, Q, 0)
    _,_,nonrob_j=inner_pgd_nonrec(pi_1, transition_matrix, cost, branch_matrix, 0.01, threshold, Q,theta)
    pi_1 = pi_now
    threshold*=0.5
    threshold = max(threshold, 1e-4)
    beta*=0.95
    non_rob.append(nonrob_j)
j_path = os.path.join(folder, "j.xlsx")
workbook = xlsxwriter.Workbook(j_path)
worksheet = workbook.add_worksheet()


for index, value in enumerate(j[1:]):
    worksheet.write(index, 0, value)
for index, value in enumerate(non_rob[1:]):
    worksheet.write(index, 1, value)

workbook.close()

