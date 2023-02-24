import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """
  伯努利多臂老虎机环境,输入K表示拉杆个数
  """

    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0~1的数，作为拉动每根拉杆的获奖概率

        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.bestprob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K  # 拉杆个数

    def step(self, k):
        """
      Input: Agent choose tht k-th arm
      Output: Return winning information according to the winning probability(probs),win 1 lose 0
      """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

#-----------------------------------------------------------------Solver注释
'''
# --------------------------------------------------------------------------------
class Solver:
    """
  Algorithm for MultiarmBandit
  1. choose action following policy
  2. get reward based on action
  3. update expectation of reward
  4. update total regret and count 
  """
    def __init__(self,bandit):
      self.bandit=bandit
      self.counts=np.zeros(self.bandit.K) # each arm's number of attemp
      self.regret=0. # the total regret until the current step (type=float)
      self.actions=[] # maintance a list -> record the action of every step
      self.regrets=[] # maintance a list -> record the regret of every step

     def update_regret(self,k):
       """
        calculate the total regret and store
         k is the number of the chosen arm
         """
       self.regret += self.bandit.best_prob - self.bandit.probs[k]
       self.regrets.append(self.regret)

  def run_one_step(self):
    """return which arm to choose
       realize 1.2.3. 
       由每个继承Solver 类的策略具体实现

    """
    raise NotImplementedError("policy not implemented")
    # https://www.bbsmax.com/A/kPzO46DQJx/

  def run(self, num_steps):
    """run several times, num_steps is the total run times"""
    for _ in range(num_steps):
      k=self.run_one_step()
      self.counts[k] += 1
      self.actions.append(k)
      self.update_regret(k)
'''
class Solver:

    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.bestprob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

# ---------------------------------------------------------------epsilon注释
'''
class EpsilonGreedy(Solver):
    """ Epsilon Greedy
  epsilon 随时间衰减

  """
  def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
    """ bandit = environment 
  """
    super().__init__(bandit)  # p1020
    #   load
    #   self.bandit=bandit;
    #   self.counts=np.zeros(self,bandit.K)
    #   self.regret=0.
    #   self.actions = []
    #   self.regrets = []
    self.epsilon = epsilon
    #   初始化拉动所有拉杆的期望奖励估值
    self.estimates = np.array([init_prob] * self.bandit.K)


  def run_one_step(self):
    if np.random.random() < self.epsilon:
        k = np.random.randint(0, self.bandit.K)  #随机选择一个拉杆
    else:
        k = np.argmax(self.estimates)  #选择期望最大的一个杆子

    r = self.bandit.step(k)  # 得到本次动作的奖励
    self.estimates[k] += [1. / (self.counts[k] + 1)] * (r - self.estimates[k])  # Q_n+1=Q_n+1/n(r_n-Q_n)
    return k
'''
class EpsilonGreedy(Solver):

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r -
                                                          self.estimates[k])
        return k

# ---------------------------------------------------------------plot注释
'''
def plot_results(solvers, solver_names):
  """生成累积懊悔随时间变化的图像。
  Input:solver是一个列表,列表中的每个元素是一个特定的策略 solver=[Solvera,Solverb,...]
  solver_name也是一个列表,储存每个策略的名称
      """
  for idx, solver in enumerate(solvers):
    time_list=range(len(solver.regrets))
    plt.plot(time_list,solver.regrets,label=solver_names[idx])
  
  plt.xlabel('Time steps')
  plt.ylabel('Cumulative regrets')
  plt.title('%d-armed bandit' %solvers[0].bandit.K)
  plt.legend()# show label
  plt.show()
'''
def plot_results(solvers:list,solver_names:list):
  for idx, solver in enumerate(solvers):
    time_list=range(len(solver.regrets))
    plt.plot(time_list,solver.regrets,label=solver_names[idx])
  plt.xlabel('Time steps')
  plt.ylabel('Cumulative regrets')
  plt.title('%d-armed bandit'%solvers[0].bandit.K)
  plt.legend()
  plt.show()

#--------------------------------------------------------------随时间衰减的epsilon
class DecayingEpsilonGreedy(Solver):
  """
  epsilon随时间衰减的epsilon=greedy
  继承Solver
  """
  def __init__(self,bandit,init_prob=1.0):
    super().__init__(bandit)
    self.estimates = np.array([init_prob]*self.bandit.K)
    self.total_count = 0  # epsilon随着total_count的变大而变小
  
  def run_one_step(self):
    self.total_count += 1
    if np.random.random()<1./self.total_count:
      k=np.random.randint(0,self.bandit.K)
    else:
      k=np.argmax(self.estimates)

    r=self.bandit.step(k)
    self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])

    return k


#-------------------------------------------------------------UpperConfidenceBound
class UCB(Solver):
  """
  1.估计拉动每根拉杆的期望奖励上界，使得拉动每根拉杆的期望奖励只有一个较小的概率p超过这个上界 
  2.选出期望上界最大的拉杆，从而选择最有可能获得最大期望奖励的拉杆
  设定概率p->Hoeffding反推U(a)->a=argmax[Q+coef*U]
  """
  def __init__(self,bandit,coef,init_prob=1.0):
    super().__init__(bandit)
    self.total_count=0
    self.estimates=np.array([init_prob]*self.bandit.K)
    self.coef=coef

  def run_one_step(self):
    self.total_count += 1
    U = self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts + 1))) # Hoeffding
    ucb=self.estimates + U
    
    k=np.argmax(ucb)
    r=self.bandit.step(k)
    self.estimates[k] += 1./(self.counts[k]+1)*(r-self.estimates[k]) # update estimate

    return k

#-------------------------------------------------------------Thompson Sampling
class ThompsonSampling(Solver):
  """
  进行一轮采样，直接选择样本中奖励最大的动作
  """
  def __init__(self,bandit):
    super().__init__(bandit)
    self._a=np.ones(self.bandit.K) # 每根拉杆奖励为1的次数
    self._b=np.ones(self.bandit.K) # 每根拉杆奖励为0的次数

  def run_one_step(self):
    samples=np.random.beta(self._a,self._b) # 利用采样建立每个arm的beta分布来近似代表arm的奖励分布，然后从这些建立的beta分布中采样一次作为评估值,这个a,b是随着后面每次采样不断更新（增加）的
    k = np.argmax(samples) # 采样奖励最大的拉杆
    r=self.bandit.step(k)  # r = 0 or 1

    self._a[k] += r
    self._b[k] += 1-r

    return k

  








