import numpy as np
from environment import inventory_control
rec_pi=np.load('ic_rec_pi.npy')
nonrec_pi=np.load('ic_nonrec_pi.npy')
print(np.round(rec_pi,2))
print(np.round(nonrec_pi,2))
max_inventory=8
rec_reward=[]
nonrec_reward=[]
max_iteration=500
env_rec=inventory_control.ic_env(max_inventory)
env_nonrec=inventory_control.ic_env(max_inventory)
for m in range(2000):
    env_rec.reset()
    env_nonrec.reset()


    for i in range(max_iteration):
        rec_state=env_rec.state
        nonrec_state=env_nonrec.state
        d=np.random.randint(0,max_inventory+1)
        env_rec.step(rec_state,np.random.choice(max_inventory + 1, p=rec_pi[rec_state]),d)
        # env_rec.step(rec_state, np.argmax(rec_pi[rec_state]), d)
        env_nonrec.step(nonrec_state,np.random.choice(max_inventory + 1, p=nonrec_pi[nonrec_state]),d)
        # env_nonrec.step(nonrec_state, np.argmax(nonrec_pi[nonrec_state]), d)
    rec_reward.append(env_rec.reward/max_iteration)
    nonrec_reward.append(env_nonrec.reward/max_iteration)
print(rec_reward)
print('rec',np.mean(rec_reward),np.std(rec_reward),np.min(rec_reward),np.max(rec_reward))
print('nonrec',np.mean(nonrec_reward),np.std(nonrec_reward),np.min(nonrec_reward),np.max(nonrec_reward))
np.save('rec_reward.npy',rec_reward)
np.save('nonrec_reward.npy',nonrec_reward)
