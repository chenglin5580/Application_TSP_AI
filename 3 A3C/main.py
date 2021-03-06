

from TSP_Burma14 import ENV
import numpy as np
import A3C_Dis as A3C
import matplotlib.pyplot as plt

env = ENV()
para = A3C.Para(env,
                units_a=300,
                units_c=300,
                MAX_GLOBAL_EP=10000,
                UPDATE_GLOBAL_ITER=1,
                gamma=0.9,
                ENTROPY_BETA=0.1,
                LR_A=0.0001,
                LR_C=0.001,
                oppo='rand_smart')
RL = A3C.A3C(para)
RL.run()
# RL.display()
