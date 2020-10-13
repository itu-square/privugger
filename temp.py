import privugger.attacker.generators as gen
import privugger.attacker.distributions as dis
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import seaborn as sns
from hypothesis import given, settings, Phase, HealthCheck, strategies as st
from functools import reduce
from sklearn.feature_selection import mutual_info_regression
import logging
l = {'BetaBinomial': [[(0.18357823453186398, ['BetaBinomial', 1, 0.001, 0.001]), (0.18648668696640203, ['BetaBinomial', 1, 0.001, 0.001])], []], 'StudentT': [[], [(0.1195985097937689, ['StudentT', 0.1, 0.0, 0.1]), (0.11676828916351752, ['StudentT', 0.1, 0.0, 0.1])]]}
for i in range(2):
    for k,v in l.items():
        print(k)
        print(v[i])
# fig, ax = plt.subplots(2,1)
# for axs in ax.flatten():
#     axs.plot(np.linspace(0,1,100), sorted(np.random.normal(1,0.1,100)))
# plt.show()