"""
Experiment 3:
    Compare IH, OH and SH results on large-scale instances
"""


from gurobipy import *
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import json
from multiprocessing import Pool
import sklearn
import scipy
from matplotlib.ticker import FuncFormatter



def random_generate_clb_d(n):
    """
    randomly generate a budgeted set
    """
    c_lb = np.random.uniform(5, 210, size=n)
    d = np.random.uniform(5, 210, size=n)
    return c_lb, d


def cal_alpha(c, c_lb, d, n, N, Gamma, start_time=0):
    """
    calculate alpha
    """
    model = Model()

    p = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='p')
    delta = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='delta')

    model.setObjective(quicksum((c_lb[i] + d[i] * delta[i]) * p[i] for i in range(n)), GRB.MAXIMIZE)

    model.addConstrs(quicksum((c[i][j] * p[i] for i in range(n))) <= 1 for j in range(N))
    model.addConstr(quicksum(delta[i] for i in range(n)) <= Gamma)
    if start_time > 0:
        model.setParam("TimeLimit", max(5, 180-(time.time()-start_time)))
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 2)
    model.optimize()

    return [delta[i].x for i in range(n)], model.ObjVal


def cal_beta(c, c_lb, d, n, N, Gamma, start_time=0):
    """
    calculate beta
    """
    res = []
    for j in range(N):
        model = Model()

        p = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='p')
        q = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='q')
        pi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='pi')

        model.setObjective(quicksum(c_lb[i] * p[i] for i in range(n)) + Gamma * pi + quicksum(q[i] for i in range(n)),
                           GRB.MINIMIZE)
        model.addConstr(quicksum(c[i][j] * p[i] for i in range(n)) >= 1)
        model.addConstrs(pi + q[i] >= d[i] * p[i] for i in range(n))
        if start_time > 0:
            model.setParam("TimeLimit", max(5, 180 - (time.time() - start_time)))
        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 2)
        model.optimize()
        res.append(model.ObjVal)

    return 1 / min(res)


def get_around_solution(c_lb, d, n, Gamma):
    """
    Generate a new solution set in the neighborhood
    """
    noise_strength = 10
    c_lb = c_lb + np.clip(np.random.normal(0, noise_strength, size=c_lb.shape), -c_lb, np.inf)
    d = d + np.clip(np.random.normal(0, noise_strength, size=d.shape), -d, np.inf)
    Gamma = Gamma + np.clip(np.random.normal(0, noise_strength), -Gamma, n - Gamma)
    return c_lb, d, Gamma


def sampling_heuristic(c, n, N, Gamma, start):
    """
    sampling heuristic process
    1. get a solution
    2. sample around it
    3. identify whether the new solution better than the solution
    4. sample around the better solution
    5. repeat
    """
    guarantee_list = []
    alpha_list = []
    beta_list = []
    Gamma_best = Gamma

    # randomly generate initial solution
    c_lb, d = random_generate_clb_d(n)
    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start)
    beta = cal_beta(c, c_lb, d, n, N, Gamma, start)

    guarantee_list.append(alpha * beta)
    alpha_list.append(alpha)
    beta_list.append(beta)
    c_lb_best = c_lb
    d_best = d

    # sampling
    start_time = time.time()
    while time.time() - start_time < 180:
        c_lb, d, Gamma = get_around_solution(c_lb_best, d_best, n, Gamma_best) # Generate a new solution in neighborhood of the best solution
        try:
            delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start_time)
            beta = cal_beta(c, c_lb, d, n, N, Gamma, start_time)
            if alpha * beta < guarantee_list[-1]:
                guarantee_list.append(alpha * beta)
                alpha_list.append(alpha)
                beta_list.append(beta)
                c_lb_best = c_lb
                d_best = d
                Gamma_best = Gamma
            else:
                guarantee_list.append(guarantee_list[-1])
                alpha_list.append(alpha_list[-1])
                beta_list.append(beta_list[-1])
        except:
            print()

    cal_time = time.time() - start
    alg_obj = get_alg_value(c_lb_best, d_best, n, N, c, Gamma)

    return guarantee_list[-1], Gamma_best, c_lb_best, d_best, alg_obj, cal_time


def get_opt_value(c, n, N):
    """
    calculate the nominal problem to get the optimal value
    """
    model = Model()
    z = model.addVar(vtype=GRB.CONTINUOUS, name='z')
    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    model.setObjective(z, GRB.MINIMIZE)
    model.addConstrs(z >= quicksum(c[i][j] * x[i] for i in range(n)) for j in range(N))
    model.addConstr(quicksum(x[i] for i in range(n)) == math.floor(n/2))
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 2)
    model.optimize()
    return model.ObjVal


def get_alg_value(c_lb, d, n, N, c, Gamma):
    """
    calculate the algorithm value
    """
    model = Model()
    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    pi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='pi')
    p = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='p')
    model.setObjective(quicksum(c_lb[i] * x[i] for i in range(n)) + Gamma * pi + quicksum(p[i] for i in range(n)))
    model.addConstrs(pi + p[i] >= d[i] * x[i] for i in range(n))
    model.addConstr(quicksum(x[i] for i in range(n)) == math.floor(n/2))
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 2)
    model.optimize()

    temp = []
    for j in range(N):
        alg_obj = 0
        for i in range(n):
            alg_obj += (c[i][j] * x[i].x)
        temp.append(alg_obj)

    return max(temp)


def inner_heuristic(c, n, N, start_time):
    """
    inner heuristic
    """
    # To get the inner budgeted set
    c_lb = []
    for i in range(n):
        c_lb.append(sum([cj for cj in c[i]])/N)

    d = []
    for i in range(n):
        d_model = Model()
        z = d_model.addVar(vtype=GRB.CONTINUOUS)
        lam = d_model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='lam')

        d_model.setObjective(z, GRB.MAXIMIZE)
        for l in range(n):
            if l != i:
                d_model.addConstr(quicksum(lam[j] * c[l][j] for j in range(N)) == c_lb[l])
        d_model.addConstr(c_lb[i] + z == quicksum(lam[j]*c[i][j] for j in range(N)))
        d_model.addConstr(quicksum(lam[j] for j in range(N)) == 1)
        d_model.setParam("OutputFlag", 0)
        d_model.setParam("Threads", 2)
        d_model.optimize()
        d.append(z.x)

    Gamma = 1

    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start_time)
    beta = cal_beta(c, c_lb, d, n, N, Gamma, start_time)
    cal_time = time.time() - start_time

    alg_obj = get_alg_value(c_lb, d, n, N, c, Gamma)

    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time


def outer_heuristic(c, n, N, start_time):
    """
    outer heuristic process
    """
    # To get the outer budgeted set
    c_lb = []
    for i in range(n):
        c_lb.append(min([cj for cj in c[i]]))
    d = []
    for i in range(n):
        d.append(max([(cj - c_lb[i]) for cj in c[i]]))
    Gamma = max([sum([(c[i][j]-c_lb[i])/d[i] for i in range(n)]) for j in range(N)])

    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start_time)
    beta = cal_beta(c, c_lb, d, n, N, Gamma, start_time)

    cal_time = time.time() - start_time
    alg_obj = get_alg_value(c_lb, d, n, N, c, Gamma)

    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time


def main(inp):
    n, N, k = inp
    res = {'sampling_guarantee': [], 'inner_guarantee': [], 'outer_guarantee': [],
           'sampling_cal_time': [], 'inner_cal_time': [], 'outer_cal_time': [],
           'N_list': [], 'n_list': [],
           'sampling_actual': [], 'inner_actual': [], 'outer_actual': [],
           'sampling_gamma': [], 'inner_gamma': [], 'outer_gamma': [],
           'sampling_c_lb':[], 'inner_c_lb': [], 'outer_c_lb': [],
           'sampling_d':[], 'inner_d': [], 'outer_d': []}


    idx = 0
    for t in range(100):
        res['N_list'].append(N)
        res['n_list'].append(n)
        idx += 1
        np.random.seed(idx)

        # using method 6 to generate dataset
        c = []
        SSIZE = 80
        Sigma = SSIZE * sklearn.datasets.make_spd_matrix(n_dim=n)
        mean = []
        temp = []
        for i in range(n):
            mean.append((25 + 100 * scipy.stats.uniform.rvs()))
        for j in range(N):
            x = scipy.stats.multivariate_normal.rvs(mean=mean, cov=Sigma)
            temp.append(x)
        for i in range(n):
            tmp = []
            for j in range(N):
                tmp.append(max(0, temp[j][i] + 300))
            c.append(tmp)

        # get the optimal value of nominal problem
        opt_obj = get_opt_value(c, n, N)

        # outer
        start = time.time()
        try:
            guarantee1, alg_obj1, c_lb1, d1, Gamma1, cal_time1 = outer_heuristic(c, n, N, start)
            res['outer_actual'].append(alg_obj1 / opt_obj)
            res['outer_guarantee'].append(guarantee1)
            res['outer_gamma'].append(Gamma1)
            res['outer_c_lb'].append(c_lb1)
            res['outer_d'].append(d1)
            res['outer_cal_time'].append(cal_time1)
        except Exception as e:
            res['outer_actual'].append(-1)
            res['outer_guarantee'].append(-1)
            res['outer_gamma'].append(-1)
            res['outer_c_lb'].append(-1)
            res['outer_d'].append(-1)
            res['outer_cal_time'].append(time.time() - start)

        # inner
        start = time.time()
        try:
            guarantee2, alg_obj2, c_lb2, d2, Gamma2, cal_time2 = inner_heuristic(c, n, N, start)
            res['inner_guarantee'].append(guarantee2)
            res['inner_actual'].append(alg_obj2 / opt_obj)
            res['inner_gamma'].append(Gamma2)
            res['inner_c_lb'].append(c_lb2)
            res['inner_d'].append(d2)
            res['inner_cal_time'].append(cal_time2)
        except Exception as e:
            res['inner_guarantee'].append(-1)
            res['inner_actual'].append(-1)
            res['inner_gamma'].append(-1)
            res['inner_c_lb'].append(-1)
            res['inner_d'].append(-1)
            res['inner_cal_time'].append(time.time() - start)

        # sampling heuristic
        start = time.time()
        Gamma = 1
        try:
            guarantee3, Gamma3, c_lb3, d3, alg_obj3, cal_time3 = sampling_heuristic(c, n, N, Gamma, start)
            res['sampling_guarantee'].append(guarantee3)
            res['sampling_actual'].append(alg_obj3 / opt_obj)
            res['sampling_gamma'].append(Gamma3)
            res['sampling_c_lb'].append(c_lb3)
            res['sampling_d'].append(d3)
            res['sampling_cal_time'].append(cal_time3)
        except Exception as e:
            res['sampling_guarantee'].append(-1)
            res['sampling_actual'].append(-1)
            res['sampling_gamma'].append(-1)
            res['sampling_c_lb'].append(-1)
            res['sampling_d'].append(-1)
            res['sampling_cal_time'].append(time.time() - start)


        with open('./res_exp3_'+str(k), 'w') as f:
            json.dump(res, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))

    return

def read_res_n3(path):
    res3 = {}
    for j in range(5, 100):
        try:
            with open(path+str(j), 'r', encoding='utf-8') as file:
                temp = json.load(file)
        except:
            continue
        for k in temp:
            if k not in res3:
                res3[k] = temp[k]
            else:
                res3[k] = res3[k]+temp[k]
    return res3

def read_res_N3(path):
    res3 = {}
    for j in range(5, 100):
        try:
            with open(path+str(100+j), 'r', encoding='utf-8') as file:
                temp = json.load(file)
        except:
            continue
        for k in temp:
            if k not in res3:
                res3[k] = temp[k]
            else:
                res3[k] = res3[k]+temp[k]
    return res3

def plot_res3(res, x_label, y_label, if_log, classes, out_path, other=''):
    rep = 100
    plt.figure()
    for c in classes:
        cc = c
        if c == 'sampling' and y_label != 'actual':
            co = 'heuristic'
        if c == 'sampling':
            co = 'red'
        elif c == 'inner':
            co = 'blue'
        elif c == 'outer':
            co = 'green'

        if c == 'inner':
            ls = '--'
        else:
            ls = '-'

        lb = c  # label
        if c == 'inner':
            lb = 'IH'
        elif c == 'outer':
            lb = 'OH'
        elif c == 'iterative':
            lb = 'IE'
        elif c == 'sampling':
            lb = 'SH'

        if if_log == 1:
            plt.plot([i for i in range(5, 100)],
                     np.log10([sum(res[cc + '_'+ y_label][rep * start:rep * start + rep]) / rep for start in range(0, 95)]),
                     color=co, label=lb)

        else:
            plt.plot([i for i in range(5, 100)],
                     [sum(res[cc + '_' + y_label][rep * start:rep * start + rep]) / rep for start in range(0, 95)],
                     color='black', label=lb, linestyle=ls)

    plt.xlabel(x_label)
    if if_log == 1:
        plt.ylabel(y_label)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(to_decimal_label))
        l = 'log'
    else:
        plt.ylabel(y_label)
        l = ''
    plt.legend()
    if x_label == 'N':
        s = 'cn'
    else:
        s = ''
    # plt.show()
    plt.savefig(out_path+y_label+'_'+other+l+s+'.pdf', bbox_inches='tight', dpi=300)

def to_decimal_label(y, pos):
    val = 10**y
    if val >= 1:
        return f'{val:.0f}'
    else:
        return f'{val:g}'

def plot_results(dir=None):
    if dir == None:
        path = './res_exp3_'
        out_path = './'
    else:
        path = './'+dir+'/res_exp3_'
        out_path = './' + dir + '/'

    res3 = read_res_n3(path)
    for y_label in ['actual', 'guarantee']: # ['guarantee', 'gamma', 'cal_time', 'actual']:
        plot_res3(res3, 'n', y_label, 0, ['inner', 'outer'], out_path, other='')
        plot_res3(res3, 'n', y_label, 1, ['sampling', 'inner', 'outer'], out_path, other='')
    rescn3 = read_res_N3(path)
    for y_label in ['actual', 'guarantee']: # ['guarantee', 'gamma', 'cal_time', 'actual']:
        plot_res3(rescn3, 'N', y_label, 0, ['inner', 'outer'], out_path, other='')
        plot_res3(res3, 'N', y_label, 1, ['sampling', 'inner', 'outer'], out_path, other='')



if __name__ == '__main__':
    # for an example
    # main([5, 100, 1])  # [n, N, k] where k is the index of result file

    # for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    # fix n = 100 and vary N in the range [5, . . . , 100],
    # separately fix N = 100 while varying n in the same range
    dataset = [(i, 100, i) for i in range(5, 100)] + [(100, i, 100+i) for i in range(5, 100)] #
    with Pool(processes=40) as pool:
        results = pool.map(main, dataset)
    plot_results()





