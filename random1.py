import numpy as np
import copy
import random
import time

##设备.#
class S():
    def __init__(self, p, t):
        self.p = p
        self.t = t
##任务.#
class T():
    def __init__(self, time, d, i, j):
        self.time = time
        self.endtime = None
        self.d = d
        self.i = i
        self.j = j
        self.turn = None
        self.ori = None
##.输入的任务队列#
class Q(list):
    def __init__(self):
        list.__init__([])
##.泳道#
class Pool(list):
    def __init__(self, deviceId, num):
        self.diviceId = deviceId
        self.num = num
        self.endtime = 0
        self.task = [0]
##评估函数.#
def evaluate(s, q1, t1, v, rtt):
    # show_p(q1,0)
    etime = 0
    q = copy.deepcopy(q1)
    t = copy.deepcopy(t1)
    size = len(s)
    pool = [0]*size
    for i in range(size):
        pool[i] = Pool()
    order = [0]*size
    for i in range(size):
        order[i] = Pool()
    timeslice = 500
    while 1:
        etime = etime +timeslice
        for y in range(0, size):

            dlist = []

            for k in range(0, len(q[y])):
                i = fpi(q[y][k])
                j = fpj(q[y][k])
                if s[y].t > 0:
                    if (j == 0 and t[i][j].ori <= etime) or (t[i][j - 1].endtime is not None and t[i][j].ori <= etime):
                        pool[y].append(q[y][k])
                        dlist.append(q[y][k])
                        order[y].append(q[y][k])
                        s[y].t = s[y].t - 1


                        if s[y].t == 0:

                            break
            for i in range(len(dlist)):
                q[y].remove(dlist[i])
            dlist.clear()
            dlist = None


        for y in range(0, size):
            # s[y].t = 0
            bb = 0
            for k in range(0, len(pool[y])):
                b1 = bb
                bb = bb + 1
                if b1 < len(pool[y]):
                    i = fpi(pool[y][b1])
                    j = fpj(pool[y][b1])

                    if j == 0 or (t[i][j - 1].endtime is not None ):
                        if t[i][j].time == t1[i][j].time:
                            if t[i][j].ori >= etime - timeslice:
                                d =  t[i][j].time - (etime - t[i][j].ori)
                            else:
                                d = t[i][j].time - timeslice
                        else:
                            d = t[i][j].time -timeslice
                        if d > 0:
                            t[i][j].time = d
                        else:
                            t[i][j].time = 0

                        if t[i][j].time == 0:
                            t[i][j].endtime = etime + d
                            # print("t[",i,"]","[",j,"]",t[i][j].ori, ':', t[i][j].endtime)
                            if j < len(t[i]) - 1:
                                # print(i)
                                # print(j)
                                xx = getx(size, q, t[i][j + 1], i, j + 1)
                                t[i][j + 1].ori = t[i][j].endtime + timedelay(t[i][j + 1], v[xx][y], rtt[xx][y])
                            del pool[y][b1]

                            bb = bb - 1
                            s[y].t = s[y].t + 1

        if num(q, size) == 0 and num(pool, size) == 0:
            #print(1)
            break

    for i in range(0, num_node):
        s[i].t = s[i].p

    total = 0
    for i in range(0, num_task):
        total = total + t[i][num_layer-1].endtime - t[i][0].ori
        # print(t[i][1].endtime)
    # print(total / 15)
    return [order, total/num_task]

def evaluate415(s, t1, v ,gene1):
    #初始化每个设备的每个泳道starttime = 0,endtime = 0#
    #循环判断每个任务t[i][j]#
    #根据t[i][j]的父任务endtime与该任务所在设备的最先空闲泳道的endtime#
    #更新t[i][j]的开始时间与结束时间并更新所运行泳道的endtime#
    #计算所有T的(endtime - starttime) / 任务数#

    t = copy.deepcopy(t1)
    genee = copy.deepcopy(gene1)

    size = len(s)
    pool = [0] * size
    for i in range(size):
        pool[i] = [0] * s[i].p
    for i in range(size):
        for j in range(s[i].p):
            pool[i][j] = Pool(i,j)

    for i in range(num_task):
        for j in range(num_layer):
            num_t = i * num_layer + j
            execution_time = Time[fpj(t[i][j])][genee[num_t]]
            father_end_time = 0
            if j == 0:
                father_end_time = t[i][0].ori
            if j != 0:
                father_end_time = t[i][j-1].endtime
                if genee[num_t] != genee[num_t-1]:
                    father_end_time = father_end_time + timedelay(t[i][j],v[j][genee[num_t]])
            execution_pool = 0
            wait = 1 #wait=1表示任务到达后需要等待
            wast_time = 100000000
            wait_time = 100000000
            for k in range(s[genee[num_t]].p):
                if pool[genee[num_t]][k].endtime <= father_end_time:
                    wait = 0
                    if father_end_time - pool[genee[num_t]][k].endtime < wast_time:
                        wast_time = father_end_time - pool[genee[num_t]][k].endtime
                        execution_pool = k
                if wait == 1:
                    if pool[genee[num_t]][k].endtime - father_end_time < wait_time:
                        wait_time = pool[genee[num_t]][k].endtime - father_end_time
                        execution_pool = k
            if wait == 0:
                t[i][j].endtime = father_end_time + execution_time
                pool[genee[num_t]][execution_pool].endtime = t[i][j].endtime
            if wait == 1:
                t[i][j].endtime = pool[genee[num_t]][execution_pool].endtime + execution_time
                pool[genee[num_t]][execution_pool].endtime = t[i][j].endtime


    #循环结束，已知所有t的结束时间
    total = 0
    for i in range(0, num_task):
        total = total + t[i][num_layer - 1].endtime - t[i][0].ori
    return total/num_task

def evaluate511(s, t1, v ,gene1):
    #初始化每个设备的每个泳道starttime = 0,endtime = 0#
    #循环判断每个任务t[i][j]#
    #根据t[i][j]的父任务endtime与该任务所在设备的最先空闲泳道的endtime#
    #更新t[i][j]的开始时间与结束时间并更新所运行泳道的endtime#
    #计算所有T的(endtime - starttime) / 任务数#

    global DeadLine
    t = copy.deepcopy(t1)
    genee = copy.deepcopy(gene1)
    totalcost = 0
    size = len(s)
    pool = [0] * size
    for i in range(size):
        pool[i] = [0] * s[i].p
    for i in range(size):
        for j in range(s[i].p):
            pool[i][j] = Pool(i,j)

    for i in range(num_task):
        for j in range(num_layer):
            num_t = i * num_layer + j
            execution_time = Time[fpj(t[i][j])][genee[num_t]]
            totalcost = totalcost + execution_time * cost_node[genee[num_t]]
            father_end_time = 0
            if j == 0:
                father_end_time = t[i][0].ori
            if j != 0:
                father_end_time = t[i][j-1].endtime
                if genee[num_t] != genee[num_t-1]:
                    father_end_time = father_end_time + timedelay(t[i][j],v[j][genee[num_t]])
            execution_pool = 0
            wait = 1 #wait=1表示任务到达后需要等待
            wast_time = 100000000
            wait_time = 100000000
            for k in range(s[genee[num_t]].p):
                if pool[genee[num_t]][k].endtime <= father_end_time:
                    wait = 0
                    if father_end_time - pool[genee[num_t]][k].endtime < wast_time:
                        wast_time = father_end_time - pool[genee[num_t]][k].endtime
                        execution_pool = k
                if wait == 1:
                    if pool[genee[num_t]][k].endtime - father_end_time < wait_time:
                        wait_time = pool[genee[num_t]][k].endtime - father_end_time
                        execution_pool = k
            if wait == 0:
                t[i][j].endtime = father_end_time + execution_time
                pool[genee[num_t]][execution_pool].endtime = t[i][j].endtime
            if wait == 1:
                t[i][j].endtime = pool[genee[num_t]][execution_pool].endtime + execution_time
                pool[genee[num_t]][execution_pool].endtime = t[i][j].endtime


    #循环结束，已知所有t的结束时间
    ct = 0
    total = 0
    for i in range(0, num_task):
        if t[i][num_layer - 1].endtime - t[i][0].ori > DeadLine:
            ct = ct + 1
        total = total + t[i][num_layer - 1].endtime - t[i][0].ori
    return [ct,totalcost]

##获取ti所属的T的编号.#
def fpi(task):
    if task is not None:
        return task.i
    return None
##获取ti在T中的序号.#
def fpj(task):
    if task is not None:
        return task.j
    return None

##数据传输时间.#
def timedelay(t, v):
    return (t.d / v) * 1000


num_node = 7    # 所有节点数：移动+边缘+云
num_task = 12   # 总任务数
num_layer = 7   # 每个任务层数
num_mobile = 4  # 移动设备个数
task_node = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0]   # 每个任务初始在哪个移动设备上
d = [1.2, 0.3, 0.8, 0.2, 0.4, 0.1, 0.05]  # 任务各层间的数据传输量
cost_node = [0, 0, 0, 0, 1.47, 1.47, 1]

s = [0] * num_node  # 所有节点集合
s[0] = S(1, 1)  # 并发数
s[1] = S(1, 1)
s[2] = S(1, 1)
s[3] = S(1, 1)
s[4] = S(4, 4)
s[5] = S(4, 4)
s[6] = S(8, 8)

t = []
for i in range(num_task):
    t.append([0] * num_layer)

for i in range(0, num_task):
    for j in range(0, num_layer):
        t[i][j] = T(1, d[j], i, j)

start = [0] * num_mobile
for j in range(0, num_mobile):
    start[j] = Q()
start[0] = [0, 4, 8, 11]    # 每个移动设备上生成的任务
start[1] = [1, 5, 9]
start[2] = [2, 6, 10]
start[3] = [3, 7]

delay = 2500              # 任务到达速
for j in range(0, num_mobile):
    c = delay/len(start[j])
    ss = 0
    for i in start[j]:
        t[i][0].ori = ss*c
        ss = ss + 1

#每层在每个节点上的执行时间
Time = np.array([[1032, 1032, 1032, 1032, 130, 130, 69],
                 [121, 121, 121, 121, 16, 16, 8],
                 [1584, 1584, 1584, 1584, 189, 189, 92],
                 [251, 251, 251, 251, 31, 31, 15],
                 [2313, 2313, 2313, 2313, 297, 297, 152],
                 [235, 235, 235, 235, 28, 28, 14],
                 [5425, 5425, 5425, 5425, 677, 677, 330]])
Time = Time / 4
#资源节点之间的传输速率
v = np.array([[100000, 0.001, 0.001, 0.001, 0.001, 10, 0.5],
              [0.001, 100000, 0.001, 0.001, 10, 10, 0.5],
              [0.001, 0.001, 100000, 0.001, 10, 10, 0.5],
              [0.001, 0.001, 0.001, 100000, 10, 0.001, 0.5],
              [0.001, 10, 10, 10, 100000, 0.001, 0.5],
              [10, 10, 10, 0.001, 0.001, 100000, 0.5],
              [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 100000]])

q = [0] * num_node
for j in range(0, num_node):
    q[j] = Q()

a = Q()
for i in range(0, num_task):
    for j in range(0, num_layer):
        a.append(t[i][j])

# 种族大小
pop_size = 100
end_size = num_node
task_size = num_task*num_layer
# 各任务可连接的节点对象       ################################记得修改
maps = {0: [0, 5, 6],
        1: [1, 4, 5, 6],
        2: [2, 4, 5, 6],
        3: [3, 4, 6],
        4: [0, 5, 6],
        5: [1, 4, 5, 6],
        6: [2, 4, 5, 6],
        7: [3, 4, 6],
        8: [0, 5, 6],
        9: [1, 4, 5, 6],
        10: [2, 4, 5, 6],
        11: [0, 5, 6]}

#粒子
class Chromosome():
    def __init__(self):
        self.gene = [0] * task_size
        for i in range(0, task_size):
            choose_pos = random.randint(0, len(maps[i // num_layer]) - 1)
            self.gene[i] = maps[i // num_layer][choose_pos]
        self.time = 0
        self.cTask = 0

def Ran():
    global bst
    global best_time
    for i in range(1000):
        for j in range(100):
            a = Chromosome()
            for k in range(0, num_task):
                a.gene[k * num_layer] = task_node[k]
                for p in range(1, num_layer):
                    if a.gene[p+(k*7)] < a.gene[p+(k*7)-1]:
                        a.gene[p+(k*7)] = a.gene[p+(k*7)-1]
            [a.cTask, a.time] = evaluate511(s, t, v, a.gene)
            if a.time < best_time and a.cTask == 0:
                best_time = a.time
                bst = copy.deepcopy(a.gene)
        print(bst)
        print(best_time)
        print("----------------------------")

bst = None
best_time = 10000000000
best_order = None

generation = 1
population = [0] * pop_size
DeadLine = 1000

g = [0, 0, 0, 0, 0, 0, 6,
     1, 1, 1, 1, 1, 1, 6,
     2, 2, 2, 2, 2, 2, 6,
     3, 3, 3, 3, 3, 3, 6,
     0, 0, 0, 0, 0, 0, 6,
     1, 1, 1, 1, 1, 1, 6,
     2, 2, 2, 2, 2, 2, 6,
     3, 3, 3, 3, 3, 3, 6,
     0, 0, 0, 0, 0, 0, 6,
     1, 1, 1, 1, 1, 1, 6,
     2, 2, 2, 2, 2, 2, 6,
     0, 0, 0, 0, 0, 0, 6]
g2 = [0, 6, 6, 6, 6, 6, 6,
     1, 6, 6, 6, 6, 6, 6,
     2, 6, 6, 6, 6, 6, 6,
     3, 6, 6, 6, 6, 6, 6,
     0, 6, 6, 6, 6, 6, 6,
     1, 6, 6, 6, 6, 6, 6,
     2, 6, 6, 6, 6, 6, 6,
     3, 6, 6, 6, 6, 6, 6,
     0, 6, 6, 6, 6, 6, 6,
     1, 6, 6, 6, 6, 6, 6,
     2, 6, 6, 6, 6, 6, 6,
     0, 6, 6, 6, 6, 6, 6]
g3 = [0, 5, 5, 5, 5, 5, 5,
     1, 5, 5, 5, 5, 5, 5,
     2, 5, 5, 5, 5, 5, 5,
     3, 4, 4, 4, 4, 4, 4,
     0, 5, 5, 5, 5, 5, 5,
     1, 5, 5, 5, 5, 5, 5,
     2, 5, 5, 5, 5, 5, 5,
     3, 4, 4, 4, 4, 4, 4,
     0, 5, 5, 5, 5, 5, 5,
     1, 5, 5, 5, 5, 5, 5,
     2, 5, 5, 5, 5, 5, 5,
     0, 5, 5, 5, 5, 5, 5]
ts = time.time()
Ran()
te = time.time()
print('time cost', te - ts, 's')

# print(evaluate511(s,t,v,rtt,g))
print("云 ：",evaluate511(s,t,v,g2))
print("边缘 ：",evaluate511(s,t,v,g3))
print("本地-云 ：",evaluate511(s,t,v,g))