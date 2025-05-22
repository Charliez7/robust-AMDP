import time

import numpy as np
import pandas as pd
import cvxpy as cp
import numpy as np
import gurobipy as grb


def generate_dirichlet_rounded(alpha, n, decimals=2):
    """
    Generate an n-length Dirichlet vector with a sum of 1.00,
    rounded to the specified decimal places.

    Parameters:
        alpha (list): Dirichlet concentration parameters.
        n (int): Length of the vector.
        decimals (int): Number of decimal places to round.

    Returns:
        np.array: Rounded Dirichlet vector summing to 1.00.
    """
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
    return np.array([generate_dirichlet_rounded([1] * n, n, decimals=2) for _ in range(m)])


def j_value_computation(d, p, pi, cost):
    """
    Compute the J value for the given parameters.

    Parameters:
        d (np.array): Initial distribution vector.
        p (np.array): Transition probability matrix.
        pi (np.array): Policy matrix.
        cost (np.array): Cost matrix.

    Returns:
        float: Computed J value.
    """
    # Compute the expected cost for each state-action pair
    expected_cost = np.sum(pi * np.sum(p * cost, axis=2), axis=1)

    # Compute the J value
    j_value = np.dot(d, expected_cost)

    return j_value, expected_cost


def v_value_computation(d, p, pi, cost):
    s_num = p.shape[0]
    P_pi = np.einsum('sa,san->sn', pi, p)
    inverse_matrix = np.linalg.inv(np.eye(s_num) - P_pi + np.tile(d, (s_num, 1)))
    c = np.einsum('sa,san->s', pi, np.einsum('san,san->san', p, cost))
    return np.dot(np.dot(inverse_matrix, np.eye(s_num) - np.tile(d, (s_num, 1))), c)


def occupancy_measure(pi, transition_matrix):
    s_num, a_num = pi.shape
    P_pi = np.zeros((s_num, s_num))
    for s in range(s_num):
        for a in range(a_num):
            P_pi[s] += pi[s, a] * transition_matrix[s, a]
    A = np.vstack([P_pi.T - np.eye(s_num), np.ones(s_num)])  # Shape (S + 1, S)
    b = np.zeros(s_num + 1)
    b[-1] = 1
    d_pi = np.linalg.lstsq(A, b, rcond=None)[0]
    time.sleep(0.1)
    return d_pi


def inner_gradient(s_num, a_num, pi, cost, d, j, v):
    gradient_matrix = np.zeros((s_num, a_num, s_num))
    for s in range(s_num):
        for a in range(a_num):
            for s_next in range(s_num):
                gradient_matrix[s, a, s_next] = d[s] * pi[s, a] * (cost[s, a, s_next] - j + v[s_next])
    return gradient_matrix


def inner_pgd(kappa, pi, transition_matrix, cost, branch_matrix, beta, threshold, type):
    s_num, a_num, _ = transition_matrix.shape
    p_old = transition_matrix.copy()
    p_new = np.zeros_like(transition_matrix)

    d = occupancy_measure(pi, p_old)
    j_current, _ = j_value_computation(d, p_old, pi, cost)
    v_current = v_value_computation(d, p_old, pi, cost)
    if type == 'sa':
        ones_s = np.ones(s_num)
        zeros_s = np.zeros(s_num)
        while True:
            j_prev = j_current
            grad_matrix = inner_gradient(s_num, a_num, pi, cost, d, j_prev, v_current)

            for s in range(s_num):
                for a in range(a_num):
                    P_sa = cp.Variable(s_num)
                    y = cp.Variable(s_num)
                    cons = [
                        P_sa @ ones_s == 1,
                        P_sa >= zeros_s,
                        ones_s @ y <= kappa[s, a],
                        P_sa <= transition_matrix[s, a] + y,
                        P_sa >= transition_matrix[s, a] - y,
                        P_sa @ branch_matrix[s, a] == 0
                    ]
                    prob = cp.Problem(
                        cp.Minimize(cp.sum_squares(P_sa - p_old[s, a]) - 2 * beta * (P_sa @ grad_matrix[s, a])),
                        cons)
                    prob.solve(solver=cp.GUROBI)
                    p_new[s, a] = P_sa.value

            d_new = occupancy_measure(pi, p_new)
            v_next = v_value_computation(d_new, p_new, pi, cost)
            j_current, _ = j_value_computation(d_new, p_new, pi, cost)
            beta *= 0.95
            if abs(j_current - j_prev) <= threshold:
                return p_new, v_next, j_current

            p_old = p_new
            v_current = v_next
            d = d_new
    if type == 's':
        ones_s = np.ones(s_num)
        zeros_s = np.zeros(a_num * s_num)
        while True:
            j_prev = j_current
            grad_matrix = inner_gradient(s_num, a_num, pi, cost, d, j_prev, v_current)
            for s in range(s_num):
                # for a in range(a_num):
                P_sa = cp.Variable(a_num * s_num)
                p_old_flatten = p_old[s].flatten()
                grad_matrix_s = grad_matrix[s].flatten()
                trans = transition_matrix[s].flatten()
                brans = branch_matrix[s].flatten()
                y = cp.Variable(a_num * s_num)
                cons = [
                    # P_sa @ ones_s == 1,
                    P_sa >= zeros_s,
                    cp.sum(y) <= kappa[s],
                    P_sa <= trans + y,
                    P_sa >= trans - y,
                    P_sa @ brans == 0
                ]
                for a in range(a_num):
                    cons.append(P_sa[a * s_num:(a + 1) * s_num] @ ones_s == 1)
                prob = cp.Problem(
                    cp.Minimize(cp.sum_squares(P_sa - p_old_flatten) - 2 * beta * (P_sa @ grad_matrix_s)),
                    cons)
                prob.solve(solver=cp.GUROBI)

                p_new[s] = P_sa.value.reshape((a_num, s_num))

            d_new = occupancy_measure(pi, p_new)
            v_next = v_value_computation(d_new, p_new, pi, cost)
            j_current, _ = j_value_computation(d_new, p_new, pi, cost)
            beta *= 0.95
            if abs(j_current - j_prev) <= threshold:
                return p_new, v_next, j_current

            p_old = p_new
            v_current = v_next
            d = d_new


def outer_gradient(s_num, a_num, transition_matrix, cost, d, j, v):
    gradient_matrix = np.zeros((s_num, a_num))
    for s in range(s_num):
        for a in range(a_num):
            # gradient_matrix[s, a] = np.dot(d, cost[s, a]) - np.dot(d, v)
            gradient_matrix[s, a] += d[s] * np.dot(transition_matrix[s, a], cost[s, a] - j + v)
    return gradient_matrix


def outer_pgd(kappa, pi_init, transition_matrix, cost, branch_matrix, beta, threshold, type):
    s_num, a_num = transition_matrix.shape[:2]
    pi_new = np.zeros((s_num, a_num))
    ones_a = np.ones(a_num)
    zeros_a = np.zeros(a_num)

    worst_transition_matrix, inner_v, j = inner_pgd(kappa, pi_init, transition_matrix, cost, branch_matrix, beta,
                                                    threshold, type)
    d = occupancy_measure(pi_init, worst_transition_matrix)

    grad = outer_gradient(s_num, a_num, worst_transition_matrix, cost, d, j, inner_v)

    for i in range(s_num):
        pi_s = cp.Variable(a_num)
        cons = [
            pi_s @ ones_a == 1,
            pi_s >= zeros_a
        ]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(pi_s - pi_init[i]) + 2 * beta * pi_s @ grad[i]), cons)
        prob.solve(solver=cp.GUROBI)
        pi_new[i] = pi_s.value
    d = occupancy_measure(pi_new, worst_transition_matrix)
    v = v_value_computation(d, worst_transition_matrix, pi_new, cost)
    _, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)

    v_s = np.zeros(s_num)
    for s in range(s_num):
        q_s = np.zeros(a_num)
        for a in range(a_num):
            q_s[a] += np.dot(worst_transition_matrix[s, a], cost[s, a] - np.dot(outer_value, d) + v)
        v_s[s] += np.dot(q_s, pi_new[s])
    j, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)
    return pi_new, outer_value, j

def rvi_sa(p, kappa, cost, p_init, branch_matrix):
    s_num = p.shape[0]
    a_num = p.shape[1]
    v_new = np.ones(s_num)
    v_old = np.zeros(s_num)
    w = v_new - v_old
    t = 0
    while np.linalg.norm(w, ord=2) >= 0.001 / 2:
        v_old = v_new.copy()
        gamma = (t + 1) / (t + 2)
        t += 1
        for s in range(s_num):
            mu = []
            for a in range(a_num):

                mid = (1 - gamma) * cost[s,a] + gamma * v_old
            # mid_cost = mid.flatten()
                trans = p[s,a]
                brans = branch_matrix[s,a]
                p_sa = cp.Variable(s_num)
                y = cp.Variable(s_num)
                cons = []


                cons.append(p_sa <= trans + y)
                cons.append(p_sa >= trans - y)
                cons.append(p_sa @ brans == 0)
                cons.append(cp.sum(y) <= kappa[s,a])
                cons.append(cp.sum(p_sa)==1)
                cons.append(y >= 0)
                prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(p_sa, mid))), cons)
                prob.solve(solver=cp.GUROBI)
                p_new = p_sa.value
                mu.append(np.dot(p_sa.value, mid))
            v_new[s] = min(mu)
        w = - v_new + v_old

    return np.dot(p_init, v_new)

def rvi_s(p, kappa, cost, p_init, branch_matrix):
    s_num = p.shape[0]

    a_num = p.shape[1]
    v_new = np.ones(s_num)
    v_old = np.zeros(s_num)
    w = v_new - v_old
    t = 0
    while np.linalg.norm(w, ord=2) >= 0.001 / 2:

        v_old = v_new.copy()
        gamma = (t + 1) / (t + 2)
        t += 1
        for s in range(s_num):
            mid = (1 - gamma) * cost[s] + gamma * v_old
            mid_cost = mid.flatten()
            trans = p[s].flatten()
            brans = branch_matrix[s].flatten()
            p_s = cp.Variable(a_num * s_num)
            y = cp.Variable(a_num * s_num)
            cons = []
            cons.append(p_s <= trans + y)
            cons.append(p_s >= trans - y)
            cons.append(p_s @ brans == 0)
            cons.append(cp.sum(y) <= kappa[s])
            cons.append(y>=0)
            for a in range(a_num):
                cons.append(p_s[a * s_num:(a + 1) * s_num] @ np.ones(s_num) == 1)
                cons.append(p_s[a * s_num:(a + 1) * s_num] >= 0)


            prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(p_s, mid_cost))), cons)
            prob.solve(solver=cp.GUROBI)
            p_new = p_s.value.reshape((a_num, s_num))
            mu = []
            for a in range(a_num):
                mu.append(np.dot(p_new[a], mid[a]))
            # sorted_index = np.argsort(-np.array(mu)).reshape(a_num)
            v_new[s] = min(mu)
        w = - v_new + v_old
            # print(v_new[s])
    # print(v_new)
    print(np.dot(p_init, v_new))
    return np.dot(p_init, v_new)


def initialize_transition_matrix(s_num, a_num):
    """
    Initialize a transition matrix with random probabilities.

    Parameters:
        s_num (int): Number of states.
        a_num (int): Number of actions.

    Returns:
        np.array: Initialized transition matrix of shape (s_num, a_num, s_num).
    """
    transition_matrix = np.zeros((s_num, a_num, s_num))
    for s in range(s_num):
        for a in range(a_num):
            transition_matrix[s, a, :] = np.random.dirichlet(np.ones(s_num))
    return transition_matrix


def inner_pgd_original_nonrec(pi, transition_matrix, cost, branch_matrix, beta, threshold, Q, theta):
    s_num, a_num, _ = transition_matrix.shape
    p_old = initialize_transition_matrix(s_num, a_num)
    d = occupancy_measure(pi, p_old)
    shape = transition_matrix.shape
    j_log = []
    j_current, _ = j_value_computation(d, p_old, pi, cost)
    j_log.append(j_current)
    v_current = v_value_computation(d, p_old, pi, cost)
    branch_vec = branch_matrix.flatten()
    while True:
        j_prev = j_current
        grad_matrix = inner_gradient(s_num, a_num, pi, cost, d, j_prev, v_current)
        grad_vec = grad_matrix.flatten()
        p_vec = cp.Variable(s_num * a_num * s_num)
        transition_vec = transition_matrix.flatten()
        p0_vec = p_old.flatten() + beta * grad_vec

        objective = cp.Minimize(cp.sum_squares(p_vec - p0_vec))
        constraints = []

        constraints.append(cp.quad_form(p_vec - transition_vec, Q) <= theta)
        constraints.append(p_vec >= 0)
        constraints.append(p_vec @ branch_vec == 0)

        for i in range(s_num):
            for j in range(a_num):

                start_idx = i * (a_num * s_num) + j * s_num
                end_idx = start_idx + s_num
                constraints.append(cp.sum(p_vec[start_idx:end_idx]) == 1)

        prob = cp.Problem(objective, constraints)


        prob.solve(solver=cp.GUROBI)
        p_new = p_vec.value.reshape(shape, order='C')



        d_new = occupancy_measure(pi, p_new)
        v_next = v_value_computation(d_new, p_new, pi, cost)
        j_current, _ = j_value_computation(d_new, p_new, pi, cost)
        # beta *= 0.95
        if abs(j_current - j_prev) <= threshold:
            return p_new, v_next, j_current

        p_old = p_new
        v_current = v_next
        d = d_new



def inner_pgd_nonrec(pi, transition_matrix, cost, branch_matrix, beta, threshold, Q, theta):
    s_num, a_num, _ = transition_matrix.shape

    p_old = transition_matrix.copy()
    shape = transition_matrix.shape
    d = occupancy_measure(pi, p_old)
    j_log = []
    j_current, _ = j_value_computation(d, p_old, pi, cost)
    j_log.append(j_current)
    v_current = v_value_computation(d, p_old, pi, cost)
    mean_w = np.zeros(s_num * a_num * s_num)
    covariance_w = np.eye(s_num * a_num * s_num)
    branch_vec = branch_matrix.flatten()

    tau = 1e5
    if theta == 0:
        return p_old, v_current, j_current
    else:
        while True:
            j_prev = j_current
            grad_matrix = inner_gradient(s_num, a_num, pi, cost, d, j_prev, v_current)
            grad_vec = grad_matrix.flatten()
            p_vec = cp.Variable(s_num * a_num * s_num)
            transition_vec = transition_matrix.flatten()

            p0_vec = p_old.flatten() + beta * grad_vec + np.sqrt(2 * beta / tau) * np.random.multivariate_normal(mean_w,
                                                                                                                 covariance_w)

            objective = cp.Minimize(cp.sum_squares(p_vec - p0_vec))
            constraints = [
                cp.quad_form(p_vec - transition_vec, Q) <= theta,
                p_vec >= 0,
                p_vec @ branch_vec == 0
            ]

            for s in range(s_num):
                for a in range(a_num):
                    start_idx = s * (a_num * s_num) + a * s_num
                    end_idx = start_idx + s_num
                    constraints.append(cp.sum(p_vec[start_idx:end_idx]) == 1)

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.GUROBI)
            p_new = p_vec.value.reshape(shape, order='C')

            d_new = occupancy_measure(pi, p_new)
            v_next = v_value_computation(d_new, p_new, pi, cost)
            j_current, _ = j_value_computation(d_new, p_new, pi, cost)

            if abs(j_current - j_prev) <= threshold:
                return p_new, v_next, j_current
            tau *= 5
            p_old = p_new
            v_current = v_next
            d = d_new

    # return j_log


def outer_pgd_nonrec(pi_init, transition_matrix, cost, branch_matrix, beta, threshold, Q, theta):
    s_num, a_num = transition_matrix.shape[:2]
    pi_new = np.zeros((s_num, a_num))
    # ones_a = np.ones(a_num)
    # zeros_a = np.zeros(a_num)

    worst_transition_matrix, inner_v, j = inner_pgd_nonrec(pi_init, transition_matrix, cost, branch_matrix, beta,
                                                           threshold, Q, theta)
    d = occupancy_measure(pi_init, worst_transition_matrix)

    grad = outer_gradient(s_num, a_num, worst_transition_matrix, cost, d, j, inner_v)

    for i in range(s_num):
        pi_s = cp.Variable(a_num)
        cons = [
            pi_s @ np.ones(a_num) == 1,
            pi_s >= np.zeros(a_num)
        ]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(pi_s - pi_init[i]) + 2 * beta * pi_s @ grad[i]), cons)
        prob.solve(solver=cp.GUROBI)
        pi_new[i] = pi_s.value
    d = occupancy_measure(pi_new, worst_transition_matrix)
    v = v_value_computation(d, worst_transition_matrix, pi_new, cost)
    _, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)

    v_s = np.zeros(s_num)
    for s in range(s_num):
        q_s = np.zeros(a_num)
        for a in range(a_num):
            q_s[a] += np.dot(worst_transition_matrix[s, a], cost[s, a] - np.dot(outer_value, d) + v)
        v_s[s] += np.dot(q_s, pi_new[s])

    # print(d)
    # print(np.dot(outer_value,d))
    j, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)
    return pi_new, outer_value, j


def outer_pgd_original_nonrec(pi_init, transition_matrix, cost, branch_matrix, beta, threshold, Q, theta):
    s_num, a_num = transition_matrix.shape[:2]
    pi_new = np.zeros((s_num, a_num))

    worst_transition_matrix, inner_v, j = inner_pgd_original_nonrec(pi_init, transition_matrix, cost, branch_matrix,
                                                                    beta,
                                                                    threshold, Q, theta)
    d = occupancy_measure(pi_init, worst_transition_matrix)

    grad = outer_gradient(s_num, a_num, worst_transition_matrix, cost, d, j, inner_v)

    for i in range(s_num):
        pi_s = cp.Variable(a_num)
        cons = [
            pi_s @ np.ones(a_num) == 1,
            pi_s >= np.zeros(a_num)
        ]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(pi_s - pi_init[i]) + 2 * beta * pi_s @ grad[i]), cons)
        prob.solve(solver=cp.GUROBI)
        pi_new[i] = pi_s.value
    d = occupancy_measure(pi_new, worst_transition_matrix)
    v = v_value_computation(d, worst_transition_matrix, pi_new, cost)
    _, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)

    v_s = np.zeros(s_num)
    for s in range(s_num):
        q_s = np.zeros(a_num)
        for a in range(a_num):
            q_s[a] += np.dot(worst_transition_matrix[s, a], cost[s, a] - np.dot(outer_value, d) + v)
        v_s[s] += np.dot(q_s, pi_new[s])
    j, outer_value = j_value_computation(d, worst_transition_matrix, pi_new, cost)
    return pi_new, outer_value, j
