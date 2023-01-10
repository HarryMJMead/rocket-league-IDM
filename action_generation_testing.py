import numpy as np

np.set_printoptions(precision=3)

def generate_action(on_ground):
    continuous_action = np.random.rand(5)*2 - 1
    if on_ground:
        continuous_action[2:5] = np.zeros(3)
    else:
        continuous_action[1] = 0
    
    jump = np.random.rand(1) > 0.99

    boost_drift = np.random.rand(2) > 0.5

    return np.concatenate((continuous_action, jump, boost_drift))

for x in range(100):
    print(generate_action(0))
