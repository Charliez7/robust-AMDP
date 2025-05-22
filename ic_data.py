import numpy as np
from environment import inventory_control

max_inventory = 8
state_num = 2 * max_inventory + 1
action_num = max_inventory + 1
env = inventory_control.ic_env(max_inventory)


def calculate_sas_arrays(state_num, action_num, max_inventory):
    transition_p_sas = np.zeros([state_num, action_num, state_num])
    validation = np.zeros([state_num, action_num, state_num])
    for s in range(state_num):
        for a in range(action_num):
            for _ in range(5):
                d1 = np.random.randint(0, max_inventory + 1)
                # d1 = int(np.clip(np.random.normal(loc=2.5, scale=1), 0, max_inventory + 1))
                env.step(s, a, d1)
                transition_p_sas[s, a, env.state] += 1
                d2 = np.random.randint(0, max_inventory + 1)
                # d2 = int(np.clip(np.random.normal(loc=2.5, scale=1), 0, max_inventory + 1))
                env.step(s, a, d2)
                validation[s, a, env.state] += 1

            transition_p_sas[s, a] /= np.maximum(np.sum(transition_p_sas[s, a]), 1)
            validation[s, a] /= np.maximum(np.sum(validation[s, a]), 1)

    return transition_p_sas, validation

def compute_sas(state_num, action_num, max_inventory, sample_num):

    env.reset()
    transition_p_sas = np.zeros([state_num, action_num, state_num])
    # r_sas=np.zeros([state_num,2,state_num])
    for samples in range(sample_num):
        env.reset()
        runiter=100*max_inventory
        for iters in range(runiter):
            state=env.state
            action = np.random.choice(action_num)
            d=np.random.choice(max_inventory+1)
            env.step(env.state,action,d)
            transition_p_sas[state, action, env.state] += 1
    for s in range(state_num):
        for a in range(action_num):
            if np.sum(transition_p_sas[s, a]) == 0:
                transition_p_sas[s, a] += 1 / state_num
            else:
                transition_p_sas[s, a] /= np.sum(transition_p_sas[s, a])

    return transition_p_sas
def calculate_true_sas(state_num, action_num, max_inventory):
    true_transition = np.zeros([state_num, action_num, state_num])
    r_sas = np.zeros([state_num, action_num, state_num])
    count_p_sa = np.zeros([state_num, action_num, state_num])
    for s in range(state_num):
        for a in range(action_num):
            for d in range(max_inventory + 1):
                curr = env.step(s, a, d)
                r_sas[s, a, env.state] += curr
                count_p_sa[s, a, env.state] += 1

            r_sas[s, a] /= np.maximum(count_p_sa[s, a], 1)
            true_transition[s, a] = count_p_sa[s, a] / np.maximum(np.sum(count_p_sa[s, a]), 1)

    return true_transition, r_sas


true_transition, r_sas = calculate_true_sas(state_num, action_num, max_inventory)
np.save('ic_true', true_transition)
np.save('cost', r_sas)
sample_num = 1000

# transition_p_sas, validation = calculate_sas_arrays(state_num, action_num, max_inventory)
transition_p_sas = compute_sas(state_num, action_num, max_inventory, sample_num)
np.save('ic_p_sas', transition_p_sas)
