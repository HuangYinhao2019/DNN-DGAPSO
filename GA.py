import numpy as np
import copy
import random
import time

##设备.#
class S():
    def __init__(self, p, t, type):
        self.p = p
        self.t = t
        self.type = type
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
            execution_time = Time[fpj(t[i][j])][s[genee[num_t]].type]
            totalcost = totalcost + execution_time * cost_node[s[genee[num_t]].type]
            father_end_time = 0
            if j == 0:
                father_end_time = t[i][0].ori
            if j != 0:
                father_end_time = t[i][j-1].endtime
                if genee[num_t] != genee[num_t-1]:
                    father_end_time = father_end_time + timedelay(t[i][j],v[s[genee[num_t]].type][s[genee[num_t-1]].type])
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


num_node = 26    # 所有节点数：移动+边缘+云
num_task = 40   # 总任务数
num_layer = 7   # 每个任务层数
num_mobile = 20  # 移动设备个数
task_node = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14,
             0, 1, 2, 3, 4]   # 每个任务初始在哪个移动设备上
d = [1.2, 0.3, 0.8, 0.2, 0.4, 0.1, 0.05]  # 任务各层间的数据传输量
cost_node = [0, 1.47, 1]

s = [0] * num_node  # 所有节点集合
for i in range(20):
    s[i] = S(1, 1, 0)
for i in range(20,25):
    s[i] = S(4, 4, 1)
s[25] = S(8, 8, 2)


t = []
for i in range(num_task):
    t.append([0] * num_layer)

for i in range(0, num_task):
    for j in range(0, num_layer):
        t[i][j] = T(1, d[j], i, j)

start = [0] * num_mobile
for j in range(0, num_mobile):
    start[j] = Q()
start[0] = [0, 20, 35]    # 每个移动设备上生成的任务
start[1] = [1, 21, 36]
start[2] = [2, 22, 37]
start[3] = [3, 23, 38]
start[4] = [4, 24, 39]
start[5] = [5, 25]
start[6] = [6, 26]
start[7] = [7, 27]
start[8] = [8, 28]
start[9] = [9, 29]
start[10] = [10, 30]
start[11] = [11, 31]
start[12] = [12, 32]
start[13] = [13, 33]
start[14] = [14, 34]
start[15] = [15]
start[16] = [16]
start[17] = [17]
start[18] = [18]
start[19] = [19]


delay = 5000              # 任务到达速
for j in range(0, num_mobile):
    c = delay/len(start[j])
    ss = 0
    for i in start[j]:
        t[i][0].ori = ss*c
        ss = ss + 1

#每层在每个节点上的执行时间
Time = np.array([[1032, 130, 69],
                 [121, 16, 8],
                 [1584, 189, 92],
                 [251, 31, 15],
                 [2313, 297, 152],
                 [235, 28, 14],
                 [5425, 677, 330]])
Time = Time / 4
#资源节点之间的传输速率
v = np.array([[100000, 10, 0.5],
              [10, 100000, 0.5],
              [0.5, 0.5, 100000]])

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
maps = {0: [0, 20, 21, 25],
        1: [1, 20, 21, 25],
        2: [2, 20, 21, 25],
        3: [3, 20, 21, 25],
        4: [4, 21, 22, 25],
        5: [5, 21, 22, 25],
        6: [6, 21, 22, 25],
        7: [7, 21, 22, 25],
        8: [8, 22, 23, 25],
        9: [9, 22, 23, 25],
        10: [10, 22, 23, 25],
        11: [11, 22, 23, 25],
        12: [12, 23, 24, 25],
        13: [13, 23, 24, 25],
        14: [14, 23, 24, 25],
        15: [15, 23, 24, 25],
        16: [16, 24, 20, 25],
        17: [17, 24, 20, 25],
        18: [18, 24, 20, 25],
        19: [19, 24, 20, 25],
        }

#粒子
class Chromosome():
    def __init__(self):
        self.gene = [0] * task_size
        self.p_bestgene = self.gene;
        for i in range(0, task_size):
            choose_pos = random.randint(0, len(maps[task_node[i // num_layer]]) - 1)
            self.gene[i] = maps[task_node[i // num_layer]][choose_pos]
        self.time = 0
        self.p_besttime = 1000000000
        self.cTask = 0
        self.prob = 0

    def mutation(self):
        pos = random.randint(0, task_size - 1) #随机选一位
        choose_pos = random.randint(0, len(maps[task_node[pos // num_layer]]) - 1) #随机选一个可放置节点
        self.gene[pos] = maps[task_node[pos // num_layer]][choose_pos] #变异

#交叉
def genetic(c1, c2):
    if c1 is None or c2 is None:
        return
    pos_num1 = random.randint(0, task_size - 1)
    pos_num2 = random.randint(0, task_size - 1)
    if pos_num1 > pos_num2:
        temp = pos_num2
        pos_num2 = pos_num1
        pos_num1 = temp
    for i in range(pos_num1, pos_num2):
        c1[i] = c2[i]

    return [copy.deepcopy(c1),copy.deepcopy(c2)]
#chr是一个粒子
# def gene_time(chr):
#     q = [0] * num_node   #q是队列
#     for j in range(0, num_node):
#         q[j] = Q()
#     # 遗传算法中控制各任务在指定节点上执行       ####修改
#     for k in range(0, num_task):
#         chr.gene[k*num_layer] = task_node[k]     ##每个任务第一层固定位置
#     task1 = copy.deepcopy(task)
#     for k in range(0, num_task):
#         pre = task_node[k]
#         for l in range(1, num_layer):
#             now = random.choice(task1[k])
#             chr.gene[k * num_layer + l] = now
#             if now != pre:
#                 task1[k].remove(pre)
#                 pre = now
#
#     for i in range(0, len(a)):
#         e = fpi(a[i])
#         r = fpj(a[i])
#         q[chr.gene[i]].append(a[i])
#         a[i].time = Time[r][chr.gene[i]]
#         # print()
#     #ts = time.time()
#     #[order, chr.time] = evaluate(s, q, t, v, rtt)
#     chr.time = evaluate415(s,t,v,rtt,chr.gene)
#     #te = time.time()
#     #print('time cost', te - ts, 's')
#     #show_p(order, chr.time)
#
#     return 0

##initialization.#
def pop_init():
    global population
    global best_time
    global bst
    global pst
    global g
    global g2
    global g3
    for i in range(pop_size):
        population[i] = Chromosome()
    population[0].gene = copy.deepcopy(g)
    population[1].gene = copy.deepcopy(g2)
    population[2].gene = copy.deepcopy(g3)
    for i in range(pop_size):
        for k in range(0, num_task):
            population[i].gene[k * num_layer] = task_node[k]
        [population[i].cTask,population[i].time] = evaluate511(s, t, v, population[i].gene)
        if population[i].time < population[i].p_besttime and population[i].cTask == 0:
            population[i].p_besttime = population[i].time
            population[i].p_bestgene = copy.deepcopy(population[i].gene)
            if population[i].time < best_time:
                best_time = population[i].time
                bst = copy.deepcopy(population[i].gene)
                pst = copy.deepcopy(bst)
##进化（迭代)
def pop_evolve():
    global population
    global best_time
    global bst
    global pst
    global generation
    global now
    #ts = time.time()
    population1 = [0] * pop_size
    for i in range(50):
        r1 = random.randint(0, 99)
        r2 = random.randint(0, 99)
        if random.random() < 0.3:
            [population1[i],population1[i+50]] = genetic(population[r1].gene,pst)
        else:
            [population1[i], population1[i + 50]] = genetic(population[r1].gene, population[r2].gene)
        if random.random() < mutation_rate:
            population[i].mutation()
        if random.random() < mutation_rate:
            population[i+50].mutation()
    for i in range(pop_size):
        population[i].gene = copy.deepcopy(population1[i])
        for k in range(0, num_task):
            population[i].gene[k * num_layer] = task_node[k]
        [population[i].cTask,population[i].time] = evaluate511(s, t, v, population[i].gene)
        # t_order = gene_time(population[i])
        if population[i].time < population[i].p_besttime and population[i].cTask == 0:
            population[i].p_besttime = population[i].time
            population[i].p_bestgene = copy.deepcopy(population[i].gene)
            if population[i].time < best_time:
                best_time = population[i].time
                now = generation
                bst = copy.deepcopy(population[i].gene)
    tt = 1000000
    for i in range(pop_size):
        if population[i].time < tt and population[i].cTask == 0:
            tt = population[i].time
            pst = copy.deepcopy(population[i].gene)
    #te = time.time()
    #print('time cost', te - ts, 's')
def GA():

    global mutation_rate
    global generation
    global now
    max_iter = 200
    generation = 1
    mutation_rate = m_max - (((m_max - m_min) / max_iter) * generation)

    while generation < max_iter:
        if generation - now > 20:
            break
        print("generation : ", generation)
        pop_evolve()
        generation = generation + 1
        print(bst)
        print(best_time)
        print("----------------------------")

    print(
        "bst------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #show_p(best_order, best_time)
    print(bst)
    print(best_time)

pst = None
bst = None
best_time = 10000000000
best_order = None

mutation_rate = 0
m_min = 0.2
m_max = 0.8

now = 1
generation = 1
population = [0] * pop_size
DeadLine = 1800

g = [0] * task_size
for i in range(num_task):
    for j in range(num_layer-1):
        g[i*num_layer+j] = task_node[i]
    g[i*num_layer+num_layer-1] = 25
g2 = [25] * task_size
for k in range(0, num_task):
    g2[k * num_layer] = task_node[k]
g3 = [0] * task_size
for i in range(0, num_task):
    g3[i * num_layer] = task_node[i]
    for j in range(1,num_layer):
        g3[i*num_layer+j] = int(20+(task_node[i]/4))

ts = time.time()
pop_init()
GA()
te = time.time()
print('time cost', te - ts, 's')

# print(evaluate511(s,t,v,rtt,g))
print("云 ：",evaluate511(s,t,v,g2))
print("边缘 ：",evaluate511(s,t,v,g3))
print("本地-云 ：",evaluate511(s,t,v,g))