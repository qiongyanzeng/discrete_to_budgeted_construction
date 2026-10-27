"""
Experiment 6:
    Bayesian optimization based on IH and OH solutions
    1. baye_same: means in one dimension
    2. baye_diff: means in three different dimensions

    only run baye diff, param optimization
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
from bayes_opt import BayesianOptimization
from scipy.stats import norm



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

def black_box_function(c, c_lb, d, n, N, gamma):
    delta, alpha = cal_alpha(c, c_lb, d, n, N, gamma, time.time())
    beta = cal_beta(c, c_lb, d, n, N, gamma, time.time())
    return alpha * beta

def get_baye_same_value(c_lb1, c_lb2, d1, d2, n, N, c, Gamma1, Gamma2, start_time, init_points=10, n_iter=50, seed=42):
    """
    Bayesian optimization using one lambda to control the convex combination of inner and outer solution
    :param c_lb1: c bar of outer budgeted data set
    :param c_lb2: c bar of inner budgeted data set
    :param d1: d of outer budgeted data set
    :param d2: d of inner budgeted data set
    :param n:
    :param N:
    :param c: discrete data set
    :param Gamma1: outer Gamma
    :param Gamma2: inner Gamma
    :param start_time: start time
    :param init_points:
    :param n_iter:
    :param seed:
    :return: guarantee, alg_obj, c_lb, d, gamma, cal_time (optimized)
    """

    # Define the optimization variable x, in [0, 1]
    pbounds = {'x': (0, 1)}

    def black_box_function1(x, c_lb1, c_lb2, d1, d2, Gamma1, Gamma2, n, N):
        """
        optimization target
        :return: guarantee
        """
        c_lb = [c_lb1[i] * x + c_lb2[i] * (1 - x) for i in range(n)]
        d = [d1[i] * x + d2[i] * (1 - x) for i in range(n)]
        gamma = Gamma1 * x + Gamma2 * (1 - x)
        delta, alpha = cal_alpha(c, c_lb, d, n, N, gamma, time.time())
        beta = cal_beta(c, c_lb, d, n, N, gamma, time.time())
        return alpha * beta

    def black_box_function_with_data1(x):
        """
        Because the variable is x, we need to nest a function to pass the parameter
        :param x:
        :return:
        """
        return -black_box_function1(x, c_lb1, c_lb2, d1, d2, Gamma1, Gamma2, n, N)

    optimizer = BayesianOptimization(
        f=black_box_function_with_data1,
        pbounds=pbounds,
        random_state=seed
    )
    optimizer.maximize(init_points, n_iter)

    # To get the result after optimization
    x = optimizer.max['params']['x']
    c_lb_opt = [c_lb1[i] * x + c_lb2[i] * (1 - x) for i in range(n)]
    d_opt = [d1[i] * x + d2[i] * (1 - x) for i in range(n)]
    gamma_opt = Gamma1 * x + Gamma2 * (1 - x)

    cal_time = time.time() - start_time
    alg_obj = get_alg_value(c_lb_opt, d_opt, n, N, c, gamma_opt)
    return -optimizer.max['target'], alg_obj, c_lb_opt, d_opt, gamma_opt, cal_time


def get_baye_diff_value(c_lb1, c_lb2, d1, d2, n, N, c, Gamma1, Gamma2, start, init_points=10, n_iter=50, seed=42):
    """
    Bayesian optimization using three different lambda to control the convex combination of inner and outer solution
    :param c_lb1: c bar of outer budgeted data set
    :param c_lb2: c bar of inner budgeted data set
    :param d1: d of outer budgeted data set
    :param d2: d of inner budgeted data set
    :param n:
    :param N:
    :param c: discrete data set
    :param Gamma1: outer Gamma
    :param Gamma2: inner Gamma
    :param start_time: start time
    :param init_points:
    :param n_iter:
    :param seed:
    :return: guarantee, alg_obj, c_lb, d, gamma, cal_time (optimized)
    """

    # Define the optimization variables x1, x2 and x3, in [0, 1]
    pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1)}

    def black_box_function2(x1, x2, x3, c_lb1, c_lb2, d1, d2, Gamma1, Gamma2, n, N):
        """
        optimization target
        :return: guarantee
        """
        c_lb = [c_lb1[i] * x1 + c_lb2[i] * (1 - x1) for i in range(n)]
        d = [d1[i] * x2 + d2[i] * (1 - x2) for i in range(n)]
        gamma = Gamma1 * x3 + Gamma2 * (1 - x3)
        delta, alpha = cal_alpha(c, c_lb, d, n, N, gamma, time.time())
        beta = cal_beta(c, c_lb, d, n, N, gamma, time.time())
        return alpha * beta
    def black_box_function_with_data2(x1, x2, x3):
        """
        Because the variables are x1, x2 and x3, we need to nest a function to pass the parameter
        :param x1:
        :param x2:
        :param x3:
        :return:
        """
        return -black_box_function2(x1, x2, x3, c_lb1, c_lb2, d1, d2, Gamma1, Gamma2, n, N)


    optimizer = BayesianOptimization(
        f=black_box_function_with_data2,
        pbounds=pbounds,
        random_state=seed
    )
    optimizer.maximize(init_points, n_iter)


    # To get the result after optimization
    x1, x2, x3 = optimizer.max['params']['x1'], optimizer.max['params']['x2'], optimizer.max['params']['x3']
    c_lb_opt = [c_lb1[i] * x1 + c_lb2[i] * (1 - x1) for i in range(n)]
    d_opt = [d1[i] * x2 + d2[i] * (1 - x2) for i in range(n)]
    gamma_opt = Gamma1 * x3 + Gamma2 * (1 - x3)

    cal_time = time.time() - start
    alg_obj = get_alg_value(c_lb_opt, d_opt, n, N, c, gamma_opt)
    return -optimizer.max['target'], alg_obj, c_lb_opt, d_opt, gamma_opt, cal_time


def inner_heuristic(c, n, N, start_time):
    """
    inner heuristic process
    :param c:
    :param n:
    :param N:
    :param start_time:
    :return: guarantee, alg_obj, c_lb, d, Gamma, cal_time
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
    :param c (list[list]): discrete data set
    :param n:
    :param N:
    :param start_time: start time
    :return: guarantee, alg_obj, c_lb, d, Gamma, cal_time
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
    """
    Bayesian optimization based on inner and outer solutions in one and three dimension
    :param inp (list): [n, N, k] where k is the index of result file
    :return:
    """
    n, N, k = inp

    res = {'inner_guarantee': [], 'outer_guarantee': [],
           'baye_same_guarantee': [], 'baye_diff_guarantee': [],
           'inner_cal_time': [], 'outer_cal_time': [],
           'baye_same_cal_time': [], 'baye_diff_cal_time': [],
           'N_list': [], 'n_list': [],
           'inner_actual': [], 'outer_actual': [],
           'baye_same_actual': [], 'baye_diff_actual': [],
           'inner_gamma': [], 'outer_gamma': [],
           'baye_same_gamma': [], 'baye_diff_gamma': [],
           'inner_c_lb': [], 'outer_c_lb': [],
           'baye_same_c_lb': [], 'baye_diff_c_lb': [],
           'inner_d': [], 'outer_d': [],
           'baye_same_d': [], 'baye_diff_d': []}

           # 'baye_diff_init_points': [],
           # 'baye_diff_n_iter': []}


    idx = 0  # random seed. Make each run use a different random seed, but reproducible

    for tt in range(10):
        res['N_list'].append(N)
        res['n_list'].append(n)
        # n = 100
        idx += 1
        np.random.seed(idx)
        c = []

        # using method 6 to generate dataset
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


        # bayesian optimization using same lambda
        # start = time.time()
        # try:
        #     guarantee, alg_obj3, c_lb3, d3, Gamma3, cal_time3 = get_baye_same_value(c_lb1, c_lb2, d1, d2, n, N, c, Gamma1, Gamma2, start)
        #     res['baye_same_guarantee'].append(guarantee)
        #     res['baye_same_actual'].append(alg_obj3 / opt_obj)
        #     res['baye_same_gamma'].append(Gamma3)
        #     res['baye_same_c_lb'].append(c_lb3)
        #     res['baye_same_d'].append(d3)
        #     res['baye_same_cal_time'].append(cal_time3)
        # except:
        #     res['baye_same_guarantee'].append(-1)
        #     res['baye_same_actual'].append(-1)
        #     res['baye_same_gamma'].append(-1)
        #     res['baye_same_c_lb'].append(-1)
        #     res['baye_same_d'].append(-1)
        #     res['baye_same_cal_time'].append(time.time() - start)


        # bayesian optimization using different lambda
        # for init_points in range(3, 15):
        #     for n_iter in range(10, 101, 5):
        init_points = 10
        n_iter = 50
        start = time.time()
        try:
            guarantee4, alg_obj4, c_lb4, d4, Gamma4, cal_time4 = get_baye_diff_value(c_lb1, c_lb2, d1, d2, n, N, c, Gamma1, Gamma2, start, init_points, n_iter)
            res['baye_diff_guarantee'].append(guarantee4)
            res['baye_diff_actual'].append(alg_obj4 / opt_obj)
            res['baye_diff_gamma'].append(Gamma4)
            res['baye_diff_c_lb'].append(c_lb4)
            res['baye_diff_d'].append(d4)
            res['baye_diff_cal_time'].append(cal_time4)
            # res['baye_diff_init_points'].append(init_points)
            # res['baye_diff_n_iter'].append(n_iter)
        except:
            res['baye_diff_guarantee'].append(-1)
            res['baye_diff_actual'].append(-1)
            res['baye_diff_gamma'].append(-1)
            res['baye_diff_c_lb'].append(-1)
            res['baye_diff_d'].append(-1)
            # res['baye_diff_init_points'].append(init_points)
            # res['baye_diff_n_iter'].append(n_iter)
            res['baye_diff_cal_time'].append(time.time() - start)


        with open('./res_exp6_'+str(k), 'w') as f:
            json.dump(res, f)

    return


def read_res_n6(path):
    # path1 = './res6_0901/res_exp6_0910_'
    res = {}
    n_list = []
    for j in range(5, 51):
        try:

            with open(path+str(j), 'r', encoding='utf-8') as file:
                temp = json.load(file)
            # with open(path1+str(j), 'r', encoding='utf-8') as file:
            #     temp1 = json.load(file)
            # temp['baye_diff_guarantee'] = temp1['baye_diff_guarantee']
            # temp['baye_diff_gamma'] = temp1['baye_diff_gamma']
            # temp['baye_diff_cal_time'] = [temp1['baye_diff_cal_time'][i] + temp['inner_cal_time'][i] + temp['outer_cal_time'][i] for i in range(10)]
            # temp['baye_diff_actual'] = temp1['baye_diff_actual']
            # temp['baye_diff_c_lb'] = temp1['baye_diff_c_lb']
            # temp['baye_diff_d'] = temp1['baye_diff_d']

            temp['baye_diff_cal_time'] = [temp['baye_diff_cal_time'][i] + temp['inner_cal_time'][i] + temp['outer_cal_time'][i] for i in range(len(temp['outer_cal_time']))]
            # temp['baye_same_cal_time'] = [temp['baye_same_cal_time'][i] + temp['inner_cal_time'][i] + temp['outer_cal_time'][i] for i in range(len(temp['outer_cal_time']))]
            n_list.append(j)
        except:
            continue
        for k in temp:
            if k not in res:
                res[k] = temp[k]
            else:
                res[k] = res[k]+temp[k]
    return res, n_list

def read_res_N6(path):
    res = {}
    n_list = []
    for j in range(5, 51):
        try:
            with open(path + str(100+j), 'r', encoding='utf-8') as file:
                temp = json.load(file)

            temp['baye_diff_cal_time'] = [
                temp['baye_diff_cal_time'][i] + temp['inner_cal_time'][i] + temp['outer_cal_time'][i] for i in
                range(len(temp['outer_cal_time']))]
            # temp['baye_same_cal_time'] = [
            #     temp['baye_same_cal_time'][i] + temp['inner_cal_time'][i] + temp['outer_cal_time'][i] for i in
            #     range(len(temp['outer_cal_time']))]
            n_list.append(j)
        except:
            continue
        for k in temp:
            if k not in res:
                res[k] = temp[k]
            else:
                res[k] = res[k] + temp[k]
    return res, n_list


def plot_res6(res, x_label, y_label, if_log, classes, out_path, N_list, other=''):

    plt.figure()
    for c in classes:
        cc = c
        if c == 'baye_same':
            co = 'red'
        elif c == 'baye_diff':
            co = 'purple'
        elif c == 'inner':
            co = 'blue'
        elif c == 'outer':
            co = 'green'

        co = 'black'
        if c == 'inner':
            ls = '--'
        elif c == 'outer':
            ls = '-'
        else:
            ls = '-.'

        lb = c  # label
        if c == 'inner':
            lb = 'IH'
        elif c == 'outer':
            lb = 'OH'
        elif c == 'iterative':
            lb = 'IE'
        elif c == 'sampling':
            lb = 'SH'
        elif c == 'baye_diff':
            lb = 'BH'



        if if_log == 1:
            # plt.plot([i for i in range(5, 50)],
            #          np.log10([sum(res[cc + '_'+ y_label][10 * start:10 * start + 10]) / 10 for start in range(0, 45)]),
            #          color=co, label=c + ', '+other)
            if co == 'purple' or co == 'red':
                plt.plot(N_list, np.log10([sum(
                    res[cc + '_' + y_label][res[x_label+'_list'].index(j):res[x_label+'_list'].index(j) + res[x_label+'_list'].count(j)]) /
                                  res[x_label+'_list'].count(j) if j in res[x_label+'_list'] else 0 for j in N_list]), color=co,
                         label=lb, linestyle=ls)
            else:
                plt.plot(N_list, np.log10([sum(
                    res[cc + '_' + y_label][
                    res[x_label + '_list'].index(j):res[x_label + '_list'].index(j) + res[x_label + '_list'].count(
                        j)]) /
                                           res[x_label + '_list'].count(j) if j in res[x_label + '_list'] else 0 for j
                                           in N_list]), color=co,
                         label=lb, linestyle=ls)

        else:
            if co == 'purple' or co == 'red':
                plt.plot(N_list, [sum(
                    res[cc + '_' + y_label][res[x_label+'_list'].index(j):res[x_label+'_list'].index(j) + res[x_label+'_list'].count(j)]) /
                                  res[x_label+'_list'].count(j) if j in res[x_label+'_list'] else 0 for j in N_list], color=co,
                         label=lb, linestyle=ls)

            else:
                plt.plot(N_list, [sum(
                    res[cc + '_' + y_label][res[x_label+'_list'].index(j):res[x_label+'_list'].index(j) + res[x_label+'_list'].count(j)]) /
                                  res[x_label+'_list'].count(j) if j in res[x_label+'_list'] else 0 for j in N_list], color=co,
                         label=lb, linestyle=ls)

    plt.xlabel(x_label)
    if if_log == 1:
        plt.ylabel('lg('+y_label+')')
        l = 'log'
    else:
        plt.ylabel(y_label)
        l = ''
    plt.legend()
    if x_label == 'N':
        s = 'cn'
    else:
        s = ''

    plt.savefig(out_path+y_label+'_'+other+l+s+'.pdf', bbox_inches='tight', dpi=300)


def plot_results(dir=None):
    if dir == None:
        path = './res_exp6_'
        out_path = './'
    else:
        path = './'+ dir +'/res_exp6_'
        out_path = './'+dir+'/'
    res6, n_list = read_res_n6(path)
    for y_label in ['guarantee']:  # ['guarantee', 'gamma', 'cal_time', 'actual']:
        plot_res6(res6, 'n', y_label, 0, ['inner', 'outer', 'baye_diff'], out_path, n_list, other='')
    res6, n_list = read_res_N6(path)
    for y_label in ['guarantee']:  # ['guarantee', 'gamma', 'cal_time', 'actual']:
        plot_res6(res6, 'N', y_label, 0, ['inner', 'outer', 'baye_diff'], out_path, n_list, other='')


if __name__ == '__main__':
    # for an example
    # main([50, 50, 1])  # [n, N, k] where k is the index of result file

    # for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    dataset = [(i, 50, i) for i in range(5, 51)] + [(50, i, 100+i) for i in range(5, 51)] #
    with Pool(processes=40) as pool:
        results = pool.map(main, dataset)

    plot_results()






