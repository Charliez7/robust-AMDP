import numpy as np
from tqdm import trange

from new_functions import *
import tqdm


def generate_dirichlet_rounded(alpha, n, decimals=2):
    # Step 1: Generate Dirichlet vector
    vector = np.random.dirichlet(alpha)  # Generate vector summing to 1

    # Step 2: Round to the specified decimals
    rounded_vector = np.round(vector, decimals)

    # Step 3: Adjust the last element to ensure the sum is exactly 1.00
    rounded_vector[-1] = 1.0 - np.sum(rounded_vector[:-1])
    rounded_vector[-1] = round(rounded_vector[-1], decimals)  # Ensure rounding precision

    return rounded_vector


def initial_distribution(n):
    return generate_dirichlet_rounded([1] * n, n, decimals=3)


def initial_policy(m, n):
    return np.array([generate_dirichlet_rounded([1] * n, n, decimals=3) for _ in range(m)])


transition_matrix = np.load('ic_true.npy')
branch_matrix = np.load('ic_p_sas.npy') * 0
cost = np.load('cost.npy')
s_num, a_num, _ = transition_matrix.shape
p_init = initial_distribution(s_num)

pi = initial_policy(s_num, a_num)
kappa = 0.1 * np.ones([s_num, a_num])
kappa_nonrec = 0.1
Q = np.eye(s_num * a_num * s_num)
threshold = 1
for i in trange(250):
    pi_now, v_now, another_j = outer_pgd(kappa, pi, transition_matrix, cost, branch_matrix, 0.01, threshold, 'sa')
    pi = pi_now
    threshold *= 0.95
np.save('ic_rec_pi', pi)
# print(np.dot(v_now,p_init))
pi = initial_policy(s_num, a_num)
threshold = 1
for i in trange(250):
    pi_now, v_now, another_j = outer_pgd_nonrec(pi, transition_matrix, cost, branch_matrix, 0.01, threshold, Q,
                                                kappa_nonrec)
    pi = pi_now
    threshold *= 0.95
np.save('ic_nonrec_pi', pi)
