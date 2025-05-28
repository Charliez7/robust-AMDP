import datetime
import random
import subprocess
import sys
import os
import tqdm
from tqdm import trange
from functions import *
import numpy as np
import xlsxwriter


def randkappa(n, m):
    kappa = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            kappa[i, j] = round(random.uniform(0.0, 0.3), 2)
    return kappa


# kappa=np.ones([s_num,a_num])*0.2

exc_time = []
dec_time = []

for j in trange(10):
    subprocess.run([sys.executable, 'garnet_mdp.py'])
    transition_matrix = np.load('transition_matrix.npy')
    branch_matrix = np.load('branch_matrix.npy')
    cost = np.load('cost.npy')
    p_init = np.load('p_init.npy')

    s_num, a_num, _ = transition_matrix.shape
    pi = initial_policy(s_num, a_num)
    pi_1 = pi.copy()
    kappa = randkappa(s_num, a_num)
    robust_vi = rvi_sa(transition_matrix, kappa, cost, p_init, branch_matrix)
    print(robust_vi)
    error_j = []
    error_j.append(1)

    error_dec = []
    error_dec.append(1)
    dec_start = datetime.datetime.now()
    threshold = 1

    for k in range(300):
        beta= np.sqrt(1 / (k + 1))
        pi_now, v_now, another_j_dec = outer_pgd(kappa, pi, transition_matrix, cost, branch_matrix, beta, threshold,'sa')
        pi = pi_now
        error_dec.append(abs(another_j_dec-robust_vi)/robust_vi)
        if (error_dec[-2]-error_dec[-1]) <= 0.001 and error_dec[-1] <= 0.02:
            break
        if k % 20 == 0 and (datetime.datetime.now()- dec_start).total_seconds() >=3000:
            print("Runtime exceeded the threshold. Breaking the loop.")
            break
        threshold *= 0.95
    dec_runtime = datetime.datetime.now() - dec_start
    dec_time.append(dec_runtime)
    print("dec",dec_runtime.total_seconds())
    pi = pi_1
    exc_start = datetime.datetime.now()
    beta = 0.1
    for i in range(300):
        beta = np.sqrt(1 / (i + 1))
        pi_now, v_now, another_j = outer_pgd(kappa, pi, transition_matrix, cost, branch_matrix, beta, 0.00001,'sa')
        pi = pi_now

        error_j.append(abs(another_j - robust_vi) / robust_vi)
        if abs(error_j[-1] - error_j[-2]) <= 0.001 and error_j[-1] <= 0.02:
            break
        if i % 20 == 0 and (datetime.datetime.now()- exc_start).total_seconds() >= 3000:
            print("Runtime exceeded the threshold. Breaking the loop.")
            break

    exc_runtime = datetime.datetime.now() - exc_start
    exc_time.append(exc_runtime)
    print("exc",exc_runtime.total_seconds())
folder= "runtime"
time_path=os.path.join(folder,"time.xlsx")
workbook = xlsxwriter.Workbook(time_path)
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'Excution Time')
worksheet.write('B1', 'Decomposition Time')
for i in range(10):
    worksheet.write(i + 1, 0, exc_time[i].total_seconds())
    worksheet.write(i + 1, 1, dec_time[i].total_seconds())
workbook.close()