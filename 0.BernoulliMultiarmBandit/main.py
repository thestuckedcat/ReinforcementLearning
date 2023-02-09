import numpy as np
import matplotlib.pyplot as plt
from EnvandSolver_MultiarmBandit import BernoulliBandit
from EnvandSolver_MultiarmBandit import EpsilonGreedy
from EnvandSolver_MultiarmBandit import plot_results
from EnvandSolver_MultiarmBandit import DecayingEpsilonGreedy

# ----------------------------------------------------------------Reference Example
np.random.seed(1)  # make experiment reproducible
# https://blog.csdn.net/weixin_45684362/article/details/126415226

K = 10
bandit_10_arm = BernoulliBandit(K)
print("Random generate a %d arm Bournulli Bandit" % K)
print("The most possible winning arm is %d, and the possibility is %.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.bestprob))


# ----------------------------------------single epsilon
'''
np.random.seed(0)
K=10
bandit_10_arm = BernoulliBandit(K)
epsilon_greedy_solver=EpsilonGreedy(bandit_10_arm,epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为:',epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver],["EpsilonGreedy"])
'''
#----------------------------------------different epsilon(choose possibility)
'''
np.random.seed(0)
epsilons=[1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver_list=[EpsilonGreedy(bandit_10_arm,epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
  solver.run(5000)
plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)
'''

#----------------------------------------Decaying Epsilon

np.random.seed(1)
decaying_epsilon_greedy_solver=DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon衰减的贪婪算法累积懊悔值为',decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver],['DecayingEpsilonGreedy'])
