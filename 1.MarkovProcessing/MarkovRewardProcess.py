import numpy as np
np.random.seed(0)


class MRP:
  def __init__(self,gamma:type=float,P:type=list[list[float]], rewards:type=list):
    '''
    Input:
    P: State Transfer Matrix
    rewards: reward on each state
    gamma: discount factor
    
    '''
    self.gamma = gamma
    self.P=np.array(P)
    self.rewards = rewards
# 给定一条trajectory，计算从某个起始状态开始到序列最后终止状态得到的回报
  def compute_return(self,start_index:type=int, chain:type=list):
    TotalRewards = 0
    for i in reversed(range(start_index,len(chain))):
      # G = R_t + \gamma R_t+1 + \gamma^2 R_t+2 + ......
      TotalRewards=gamma * TotalRewards + rewards[chain[i]-1] # chain里面的是状态，而列表索引是从0开始，所以匹配上需要-1
    return TotalRewards
# 环境本身每个状态的期望回报（与这个状态能转移到的状态的价值有关，是env本身的性质）
  def compute_value(self,states_num:type=int):
    '''
    Input:
    #------------------------------already in class
    P: State Transfer Matrix
    rewards: reward on each state
    gamma: discount factor
    #------------------------------new input
    states_num: state number of Markov reward process
    '''

    rewards_col = np.array(self.rewards).reshape((-1,1)) # 写为列向量
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num)-self.gamma*self.P),rewards_col) # V=(I-gammaP)^-1 R
    return value






# 定义状态转移概率矩阵 P_i,j-> possibility of state_i to state_j
P=[
  [0.9,0.1,0.0,0.0,0.0,0.0],
  [0.5,0.0,0.5,0.0,0.0,0.0],
  [0.0,0.0,0.0,0.6,0.0,0.4],
  [0.0,0.0,0.0,0.0,0.3,0.7],
  [0.0,0.2,0.3,0.5,0.0,0.0],
  [0.0,0.0,0.0,0.0,0.0,1.0],

]
rewards = [-1,-2,-2,10,1,0] # 定义奖励函数（转移到状态i时返回的奖励）
gamma=0.5 # 折扣因子
states_num = 6

MarkovProcess = MRP(gamma, P, rewards)


# define a chain s1->s2->s3->s6
chain=[1,2,3,6]
start_index = 0
G = MarkovProcess.compute_return(start_index, chain)

print("根据序列计算得到的回报为:%s。"%G)


V = MarkovProcess.compute_value(states_num)
print("MRP中的每个状态价值分别为\n",V)