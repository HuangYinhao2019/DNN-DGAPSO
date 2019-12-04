import numpy as np
import copy
import random1
import time

class S():
    def __init__(self, pool_num, free_num):
        self.pool_num = pool_num
        self.free_num = free_num


class T():
    def __init__(self, time, d, i, j):
        self.time = time
        self.endtime = None
        self.d = d
        self.i = i
        self.j = j
        self.turn = None
        self.ori = None


class make_empty_list(list):

    def __init__(self):
        list.__init__([])


class Pool(list):

    def __init__(self):
        list.__init__([])


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

    while 1:
        timeslice = 200
        # 把任务放进泳道
        for y in range(0, size):
            dlist = []
            # 遍历每个节点上泳道中的任务
            for k in range(0, len(q[y])):
                # 获取泳道中任务的编号跟层号
                i = fpi(q[y][k])
                j = fpj(q[y][k])
                # 如果有空闲泳道
                if s[y].free_num > 0:
                    # 如果是第一层且已经到达或者是前一层任务已完成且当前层已经到达
                    if (j == 0 and t[i][j].ori <= etime) or (t[i][j - 1].endtime is not None and t[i][j].ori <= etime):
                        pool[y].append(q[y][k])
                        # 临时记录已经加入泳道的任务
                        dlist.append(q[y][k])
                        # 记录任务执行的顺序
                        order[y].append(q[y][k])
                        # 空闲泳道数减一
                        s[y].free_num = s[y].free_num - 1
                        # 如果不存在空闲泳道
                        if s[y].free_num == 0:
                            break

            # 把已经加入泳道的任务从列表移除
            for i in range(len(dlist)):
                q[y].remove(dlist[i])
            dlist.clear()

        # 找出泳道中的最小时间作为时间片
        for y in range(0, size):
            # 遍历每个节点上泳道中的任务
            for k in range(0, len(pool[y])):
                i = fpi(pool[y][k])
                j = fpj(pool[y][k])
                if task[i][j].time<timeslice:
                    timeslice=task[i][j].time

        etime = etime + timeslice
        #print(etime,timeslice)
        # 遍历所有节点
        for y in range(0, size):
            # 遍历每个节点上泳道中的任务
            bb = 0
            for k in range(0, len(pool[y])):
                b1 = bb
                bb = bb + 1
                if b1 < len(pool[y]):
                    i = fpi(pool[y][b1])
                    j = fpj(pool[y][b1])
                    # 当前层是第0层或者前一层已经执行完
                    if j == 0 or (t[i][j - 1].endtime is not None):
                        d = t[i][j].time - timeslice
                        if d > 0:
                            t[i][j].time = d
                        else:
                            t[i][j].time = 0

                        if t[i][j].time == 0:
                            t[i][j].endtime = etime
                            # print("t[",i,"]","[",j,"]",t[i][j].ori, ':', t[i][j].endtime)
                            if j < len(t[i]) - 1:
                                # print(i)
                                # print(j)
                                # 获取下一层所在的节点
                                xx = getx(size, q, t[i][j + 1], i, j + 1)
                                if xx!=y:
                                    # 下一层任务的到达时间为当前层的完成时间+传输延迟
                                    t[i][j + 1].ori = t[i][j].endtime + timedelay(t[i][j + 1], v[xx][y], rtt[xx][y])
                                else:
                                    t[i][j + 1].ori = t[i][j].endtime
                            del pool[y][b1]

                            bb = bb - 1
                            s[y].free_num = s[y].free_num + 1

        if num(q, size) == 0 and num(pool, size) == 0:
            #print(1)
            break

    for i in range(0, num_node):
        s[i].free_num = s[i].pool_num

    total = 0
    for i in range(0, num_task):
        total = total + t[i][num_layer-1].endtime - t[i][0].ori
    return [order, total/num_task]


def num(pool, size):
    for y in range(0, size):
        for b in range(0, len(pool[y])):
            if pool[y][b] is not None:
                return 1
    return 0


def fpi(task):
    if task is not None:
        return task.i
    return None


def fpj(task):
    if task is not None:
        return task.j
    return None


def getx(size, pool, t, i, j):
    for x in range(0, size):
        for b in range(0, len(pool[x])):
            # if pool[x][b] == t:
            # if t in pool[x]:
            i1 = fpi(pool[x][b])
            j1 = fpj(pool[x][b])
            if i1 == i and j1 == j:
                return x
    return -1


def timedelay(t, v, rtt):
    global const_num
    a = (t.d / v) * const_num
    return max(a, rtt)


num_node = 7    # 所有节点数：移动+边缘+云

num_layer = 7   # 每个任务层数
num_mobile = 4  # 移动设备个数
data = [1.2, 0.3, 0.8, 0.2, 0.4, 0.1, 0.05]  # 任务各层间的数据传输量

s = [0] * num_node  # 所有节点集合
s[0] = S(1, 1)  # 并发数
s[1] = S(1, 1)
s[2] = S(1, 1)
s[3] = S(1, 1)
s[4] = S(2, 2)
s[5] = S(2, 2)
s[6] = S(8, 8)

num_task = 12   # 总任务数
# 每个节点在哪个设备上产生
start = [[0, 4, 8,  11],
         [1, 5, 9],
         [2, 6, 10],
         [3, 7]]

# 任务矩阵，task[i][j]表示第i个任务第j层
task = []
for i in range(num_task):
    task.append([0] * num_layer)


for i in range(0, num_task):
    for j in range(0, num_layer):
        task[i][j] = T(1, data[j], i, j)


const_num = 1000
delay = 2500
for j in range(0, num_mobile):
    c = delay/len(start[j])
    ss = 0
    for i in start[j]:
        task[i][0].ori = ss * c
        ss = ss + 1

# 边缘之间迁移和选择边缘的情况要记得修改染色体函数
Time = np.array([[163, 163, 163, 163, 107, 81, 69],
                 [12, 12, 12, 12, 10, 10, 8],
                 [219, 219, 219, 219, 132, 109, 92],
                 [21, 21, 21, 21, 18, 16, 15],
                 [313, 313, 313, 313, 231, 185, 152],
                 [25, 25, 25, 25, 22, 18, 14],
                 [820, 820, 820, 820, 583, 394, 330]])

# 各个节点之间的传输速率
v = np.array([[100000, 0.001, 0.001, 0.001, 0.001, 1, 0.2],
              [0.001, 100000, 0.001, 0.001, 1, 0.001, 0.2],
              [0.001, 0.001, 100000, 0.001, 1, 0.001, 0.2],
              [0.001, 0.001, 0.001, 100000, 0.001, 2, 0.2],
              [0.001, 1, 1, 0.001, 100000, 0.001, 0.3],
              [1, 0.001, 0.001, 1, 0.001, 100000, 0.3],
              [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 100000]])
# 各个节点之间的往返时间
rtt = np.array([[0, -1, -1, -1, -1, 30, 150],
                [-1, 0, -1, -1, 30, -1, 150],
                [-1, -1, 0, -1, 30, -1, 150],
                [-1, -1, -1, 0, -1, 30, 150],
                [-1, 30, 30, -1, 0, -1, 100],
                [30, -1, -1, 30, -1, 0, 100],
                [150, 150, 150, 150, 100, 100, 0]])

q = [0] * num_node
for j in range(0, num_node):
    q[j] = make_empty_list()

task_info = make_empty_list()
for i in range(0, num_task):
    for j in range(0, num_layer):
        task_info.append(task[i][j])

# 种群大小
pop_size = 100
end_size = num_node
task_size = num_task*num_layer


# 记录每个任务可连接的节点
available_node = [0] * num_task
for j in range(0, num_task):
    available_node[j] = make_empty_list()
for i in start[0]:
    available_node[i] = [0, 5, 6]
for i in start[1]:
    available_node[i] = [1, 4, 6]
for i in start[2]:
    available_node[i] = [2, 4, 6]
for i in start[3]:
    available_node[i] = [3, 5, 6]


# 随机生成染色体基因
class Chromosome:
    def __init__(self):
        self.gene = [0] * task_size
        for k in range(0, num_task):
            # 随机产生两个迁移位点
            off_pos1 = random1.randint(1, 5)
            off_pos2 = random1.randint(off_pos1 + 1, 6)
            for l in range(0,off_pos1):
                self.gene[k * num_layer + l]=available_node[k][0]
            # 此处为选择边缘情况专用，记得修改
            # 随机选择两个边缘中的一个
            # r = random.randint(1, 2)
            for l in range(off_pos1, off_pos2):
                # available_node[k][1]或者available_node[k][r]（选择边缘）
                self.gene[k * num_layer + l] = available_node[k][1]
            # 此处为边缘间迁移情况专用，记得修改
            # 最后一次迁移随机在云端和另一个边缘之间选择
            # r = random.randint(0,1)
            # if r == 0:
                # 选云端
                # r1 = 6
            # else:
                # 选另一个边缘
            #    if available_node[k][1] == 4:
            #        r1 = 5
            #    else:
            #        r1 = 4
            for l in range(off_pos2, 7):
                # available_node[k][-1]或者r1(边缘迁移)
                self.gene[k * num_layer + l] = available_node[k][-1]

        self.time = 0
        self.prob = 0

    # 突变
    def mutation(self):
        # 随机生成要变异的基因数量
        pos_num = random1.randint(0, task_size - 1)

        for i in range(pos_num):
            # 随机生成要变异的基因位点
            pos = random1.randint(0, task_size - 1)
            if pos % num_layer == 0:
                pos = pos + 1
            choose_pos = random1.randint(0, len(available_node[pos // num_layer]) - 1)
            no = pos // num_layer
            # 基因突变,把突变节点的任务后继子任务全部突变
            for j in range(pos % num_layer, num_layer):
                self.gene[no * num_layer + j] = available_node[pos // num_layer][choose_pos]


# 所有个体的总评估时间
total_time=0
# 所有个体的平均评估时间
average_time = 11111111
population = [0] * pop_size
y = 111111111
mutation_rate = 0.05
bst = None


# 打印种群
def show_p(tmp_q, stime):
    for i in range(0, len(tmp_q) - 1):
        print("{",end="")
        for j in range(0, len(tmp_q[i])-1):
            print("(", fpi(tmp_q[i][j]), ",", fpj(tmp_q[i][j]), "),", end="")
        if len(tmp_q[i])==0:
            print("},")
        elif len(tmp_q[i])==1:
            print("(", fpi(tmp_q[i][0]), ",", fpj(tmp_q[i][0]), ")},")
        else:
            print("(", fpi(tmp_q[i][j+1]), ",", fpj(tmp_q[i][j+1]), ")},")
    print("{", end="")
    for j in range(0, len(tmp_q[i+1]) - 1):
        print("(", fpi(tmp_q[i+1][j]), ",", fpj(tmp_q[i+1][j]), "),", end="")
    if len(tmp_q[i+1]) == 0:
        print("},")
    elif len(tmp_q[i+1]) == 1:
        print("(", fpi(tmp_q[i+1][0]), ",", fpj(tmp_q[i+1][0]), ")},")
    else:
        print("(", fpi(tmp_q[i+1][j + 1]), ",", fpj(tmp_q[i+1][j + 1]), ")},")
    print("迭代次数:",generation)
    print("耗时:     ", stime)


# 单点交叉
def genetic(c1, c2):
    if c1 is None or c2 is None:
        return
    if c1 is None or c2.gene is None:
        return
    if c1.gene is None or c2 is None:
        return
    if len(c1.gene) != len(c2.gene):
        return

    #  随机选择基因位点
    pos = random1.randint(0, task_size - 1)

    for i in range(pos, task_size):
        if i % num_layer != 0:
            tmp = c1.gene[i]
            c1.gene[i] = c2.gene[i]
            c2.gene[i] = tmp
    return [copy.deepcopy(c1), copy.deepcopy(c2)]


# 计算个体的评估时间
def gene_time(chr):
    q = [0] * num_node
    for j in range(0, num_node):
        q[j] = make_empty_list()

    for i in range(0, len(task_info)):
        layer = fpj(task_info[i])
        q[chr.gene[i]].append(task_info[i])
        task_info[i].time = Time[layer][chr.gene[i]]
    [order, chr.time] = evaluate(s, q, task, v, rtt)
    show_p(order, chr.time)
    # 返回方案
    return order


best_time = 1111111111
des = 1000


# 计算所有个体的平均评估时间
def pop_cal():
    global best_time, average_time, total_time, bst, bst_gene, generation
    total_time = 0
    # 找出最优个体
    for i in range(0, len(population)):
        tmp_q = gene_time(population[i])
        if population[i].time < best_time:
            # 最优个体的评估时间
            best_time = population[i].time
            # 记录最优个体的方案
            bst = copy.deepcopy(tmp_q)
            # 最优个体所在的代数
            bst_gene = generation
        total_time = total_time + population[i].time
    average_time = total_time / len(population)


# 个体的概率
def pop_p():
    for i in range(pop_size):
        # 越优良的个体越容易被选到
        population[i].prob = (1 - population[i].time / total_time) / (pop_size - 1)


# 生成初代种群
def pop_init():
    for i in range(0, pop_size):
        population[i] = Chromosome()
    # 计算所有个体的平均评估时间
    pop_cal()
    # 个体的概率
    pop_p()


# 种群变异
def pop_mutation():
    global population
    for i in range(0, len(population)):
        if random1.random1() < mutation_rate:
            population[i].mutation()


# 选择优良父代以产生子代
def get_parent():
    lucky = random1.random1()
    sum_lucky = 0
    while 1:
        i = random1.randint(0, pop_size - 1)
        if population[i].time < average_time:
            sum_lucky += population[i].prob
        if sum_lucky > lucky:
            return population[i]
    return None


# 种群进化
def pop_evolve():
    global population
    # 子代种群
    child_population = []
    # 选择优良个体保留到下一代
    for i in range(0, len(population)):
        if population[i].time < average_time:
            child_population.append(population[i])

    while len(child_population) < pop_size:
        p1 = get_parent()
        p2 = get_parent()
        if p1 != p2:
            children = genetic(p1, p2)
            if children is not None:
                for i in range(len(children)):
                    child_population.append(children[i])

    population.clear()
    population = None
    population = child_population
    pop_mutation()
    pop_cal()
    pop_p()


bst_gene = 0
generation = 1


def ga():
    time_begin = time.time()
    global generation
    max_iter = 150
    pop_init()
    while generation < max_iter:
        print("generation : ", generation)
        pop_evolve()
        generation = generation + 1
    time_end = time.time()
    cost_time = time_end - time_begin
    print(
        "bst-------------------------")
    show_p(bst, best_time)
    print("最优迭代次数：", bst_gene)
    print("最优耗时：", end=" ")
    print(cost_time)

ga()
