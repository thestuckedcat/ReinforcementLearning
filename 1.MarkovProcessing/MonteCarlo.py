import numpy as np
np.random.seed(0)

class MC:
  def __init__(self):
    self.S = ["s1","s2","s3","s4","s5"] #状态集合
    self.A = ["stay s1", "goto s1", "goto s2", "goto s3", "goto s4", "goto s5", "possibly goto"] #动作集合
    # 状态转移函数
    self.P = {
      "s1-stay s1-s1":1.0,"s1-goto s2-s2":1.0,
      "s2-goto s1-s1":1.0, "s2-goto s3-s3":1.0,
      "s3-goto s4-s4":1.0, "s3-goto s5-s5":1.0, 
      "s4-goto s5-s5":1.0, "s4-possibly goto-s2":0.2,
      "s4-possibly goto-s3":0.4, "s4-possibly goto-s4":0.4
    }
    # 奖励函数
    self.R = {
      "s1-stay s1":-1, "s1-goto s2":0,
      "s2-goto s1":-1, "s2-goto s3":-2, 
      "s3-goto s4":-2, "s3-goto s5":0,
      "s4-goto s5":10, "s4-possibly goto":1
    }

    self.gamma = 0.5
    self.MDP = (self.S,self.A,self.P,self.R,self.gamma)

    # 策略1,随机策略
    self.Pi_1 = {
      "s1-stay s1":0.5, "s1-goto s2":0.5,
      "s2-goto s1":0.5, "s2-goto s3":0.5,
      "s3-goto s4":0.5, "s3-goto s5":0.5,
      "s4-goto s5":0.5, "s4-possibly goto":0.5 
    }

    # 策略2
    self.Pi_2={
      "s1-stay s1":0.6, "s1-goto s2":0.4,
      "s2-goto s1":0.3, "s2-goto s3":0.7,
      "s3-goto s4":0.5, "s3-goto s5":0.5,
      "s4-goto s5":0.1, "s4-possibly goto":0.9
    }
  
  def join(self,str1:type=str,str2:type=str):
    #把输入的两个字符串通过-连接，便于使用上述定义的P，R
    return str1+'-'+str2

  def sample(self, MDP:type=tuple, Pi:type=dict, timestep_max:type=int, number:type=int):
    '''
    采样函数， 策略pi,限制最长时间步timestep_max, 总共采样序列数number
    '''
    S, A, P, R, gamma = MDP
    episodes = [] # episode set
    for _ in range(number): # number of sample times/number of trajectory 
      trajectory=[]
      timestep=0
      s=S[np.random.randint(4)] #随机选择一个除了s5之外的状态s作为起点

      # 当前状态为终止状态或者时间步太长时，一次采样结束
      while s != "s5" and timestep <= timestep_max:
        timestep += 1
        temp = 0
        rand = np.random.rand()
        # 在状态s下根据策略选择动作
        for a_opt in A:
          temp += Pi.get(self.join(s,a_opt), 0) # **获得策略概率，若不存在则返回0,采用累加保证一定能选到一个动作
          # rand = np.random.rand()
          if temp > rand:
            a = a_opt
            r = R.get(self.join(s,a),0)
            break

        # 根据状态转移概率得到下一个状态s_next
        

        temp = 0
        rand = np.random.rand()
        for s_opt in S:
          #print(self.join(self.join(s,a),s_opt))
          temp += P.get(self.join(self.join(s,a),s_opt),0)
          
          # rand = np.random.rand()
          # print("temp={0}, rand={1}".format(temp, rand))
          if temp > rand:
            s_next = s_opt
            #print("episode={2},s={0},s_next={1}".format(s,s_next,_))
            break

        trajectory.append((s,a,r,s_next)) # 把这一次转移元组加入序列中
        # print('episode = {}, s = {}, a = {}, r = {}, s_next = {}\n'.format(_,s,a,r,s_next))
        s = s_next # s_next变成当前状态，开始接下来的循环

      episodes.append(trajectory)
    return episodes

  def compute_episode_value(self, episodes, V, N, gamma):
    '''
    episodes: 每次采样的序列
    V : 状态价值（迭代的初始价值）
    N : 增量计数
    gamma : 衰减率
    N = N + 1
    V = V + 1/N(G-V)
    '''
    for trajectory in episodes:
      G = 0
      for i in range(len(trajectory)-1,-1,-1): # 一个序列从后往前计算
        (s,a,r,s_next)=trajectory[i]
        G = r + gamma * G
        N[s] = N[s]+1
        V[s] = V[s] + (G - V[s])/N[s]
    
  def compute_occupancy(self, episodes, s, a, timestep_max, gamma):
    '''
    计算状态动作对(s,a)出现的频率，以此来估算策略的占用度量
    '''
    rho = 0
    total_times = np.zeros(timestep_max) #记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max) # 记录(s_t, a_t)=(s,a)的次数
    for episode in episodes:
      for i in range(len(episode)):
        (s_opt, a_opt, r, s_next) = episode[i]
        total_times[i] += 1
        if s == s_opt and a == a_opt:
          occur_times[i] += 1
    for i in reversed(range(timestep_max)):
      if total_times[i]:
        rho += gamma ** i * occur_times[i] / total_times[i]

    return (1-gamma)*rho



MenteCarlo = MC()       

#-----------------------------------------采样example：采样五次，每个序列不超过20步

episodes = MenteCarlo.sample(MenteCarlo.MDP, MenteCarlo.Pi_1, 20, 5)

for _ in range(5):
  print('第{0}条序列\n'.format(_+1),episodes[_])

#-----------------------------------------状态价值example：采样1000次
timestep_max = 20
episodes = MenteCarlo.sample(MenteCarlo.MDP, MenteCarlo.Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1":0, "s2":0, "s3":0,"s4":0,"s5":0}
N = {"s1":0, "s2":0, "s3":0,"s4":0,"s5":0}
MenteCarlo.compute_episode_value(episodes, V,N,gamma)

print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)
    
#-----------------------------------------计算占用度量
gamma = 0.5
timestep_max = 1000

episodes_1 = MenteCarlo.sample(MenteCarlo.MDP, MenteCarlo.Pi_1 ,timestep_max, 1000)
episodes_2 = MenteCarlo.sample(MenteCarlo.MDP, MenteCarlo.Pi_2 ,timestep_max, 1000)

rho_1 = MenteCarlo.compute_occupancy(episodes_1, "s4", "possibly goto", timestep_max, gamma)
rho_2 = MenteCarlo.compute_occupancy(episodes_2, "s4", "possibly goto", timestep_max, gamma)
print(rho_1, rho_2)
