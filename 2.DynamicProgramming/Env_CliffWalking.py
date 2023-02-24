import copy

class CliffWalkingEnv:
  """
  悬崖漫步环境
  """
  def __init__(self, ncol=12, nrow=4):
    self.ncol = ncol
    self.nrow = nrow
    # 转移矩阵P[state][action] = [(p, next_state, reward, done)] 包含转移概率，下一个状态，奖励，是否结束,这个矩阵遍历了所有情况
    self.P = self.createP()

  def createP(self):
    # 初始化,每个地图格子拥有四个方向的评分
    P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] 
    #list Comprehensions [traget for i in range(4)]=[target target target target]

    # 4种动作，change[0]:上，change[1]:下， change[2]:左，change[3]:右 坐标系原点(0,0)
    # 定义在左上角
    change = [[0,-1],[0,1],[-1,0],[1,0]]
    for i in range(self.nrow):
      for j in range(self.ncol):
        for a in range(4):
          # 当前位置在悬崖或者目标状态， 因为无法继续交互，任何动作奖励都为0
          if i == self.nrow - 1 and j > 0:
            P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
            continue
          
          # 其他位置
          next_x = min(self.ncol-1, max(0, j+change[a][0])) # 动作既不能超过右边界(self.ncol-1)也不能超过左边界(0)
          next_y = min(self.nrow-1, max(0, i+change[a][1]))
          next_state = next_y * self.ncol + next_x          # 新转移到的位置的坐标(next_x,next_y)，因为P是线性存储的，所以这个公式计算了转移到的位置
          reward = -1
          done = False

          # 下一个位置在悬崖或者终点
          if next_y == self.nrow - 1 and next_x > 0:
            done = True
            if next_x != self.ncol - 1: #下一个位置在悬崖
              reward = -100
            P[i*self.ncol + j][a] = [(1, next_state, reward, done)]

    return P



