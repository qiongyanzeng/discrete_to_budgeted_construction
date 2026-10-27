"""
Experiment 8:
    Test and compare IH, OH and SH on Chicago travel time dataset.
    For 10 pairs of fixed s-t, three algorithms are executed respectively.
    For same N, the scenarios lists among different pairs are same.
    For larger N case, the scenarios that selected in smaller N cases are included in its selected scenarios.
"""

from gurobipy import *
import pandas as pd
import random
import time
import numpy as np
import json
from multiprocessing import freeze_support, Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



def shortest_path_discrete(data, nodes, s, t, scenario_cols):
    """
    Solve the shortest path problem under discrete uncertainty
    :param data (dataframe):
    :param nodes (list): the list of all nodes
    :param s (int): start node
    :param t (int): end node
    :param scenario_cols (list): selected scenarios
    :return: objective value (float), edges number in the shortest path (int)
    """
    model = Model("RobustShortestPathDiscrete")
    x = model.addVars(data.index, vtype=GRB.BINARY, name="x")
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    for scen in scenario_cols:
        model.addConstr(
            quicksum(data.loc[k, scen] * x[k] for k in data.index) <= z
        )

    model.setObjective(z, GRB.MINIMIZE)

    for i in nodes:
        model.addConstr(
            quicksum(x[k] for k in data.index if data.loc[k, "start_node"] == i) -
            quicksum(x[k] for k in data.index if data.loc[k, "end_node"] == i)
            ==
            (1 if i == s else (-1 if i == t else 0))
        )

    model.optimize()
    return model.ObjVal, sum([x[k].x for k in data.index])



def cal_alpha(c, c_lb, d, n, N, Gamma, start_time=0):
    """
    calculate alpha
    :param c: discrete data set
    :param c_lb: budgeted data set
    :param d:
    :param n:
    :param N:
    :param Gamma:
    :param start_time: Calculation start time. If it is 0, there is no timelimit, otherwise the calculation time is limited to 3 minutes
    :return: deltas' values (list), alpha value (float)
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
    :param c: discrete data set
    :param c_lb: c bar of budgeted data set
    :param d: d of budgeted data set
    :param n:
    :param N:
    :param Gamma:
    :param start_time: Calculation start time. If it is 0, there is no timelimit, otherwise the calculation time is limited to 3 minutes
    :return: beta value (float)
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


def get_alg_value(c_lb, d, N, c, Gamma, s, t, nodes, data):
    """
    calculate the algorithm value
    :param c_lb (list): c bar of budgeted data set
    :param d (list): d of budgeted data set
    :param N (int):
    :param c (list[list]): discrete data set
    :param Gamma (float):
    :param s (int): start node
    :param t (int): end node
    :param nodes (list): the list of all nodes
    :param data (dataframe):
    :return: the algorithm value (int)
    """
    model = Model()
    x = model.addVars(data.index, vtype=GRB.BINARY, name='x')
    pi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='pi')
    p = model.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, name='p')
    model.setObjective(quicksum(c_lb[k] * x[k] for k in data.index) + Gamma * pi + quicksum(p[k] for k in data.index))
    model.addConstrs(pi + p[k] >= d[k] * x[k] for k in data.index)

    for i in nodes:
        model.addConstr(
            quicksum(x[k] for k in data.index if data.loc[k, "start_node"] == i) -
            quicksum(x[k] for k in data.index if data.loc[k, "end_node"] == i)
            ==
            (1 if i == s else (-1 if i == t else 0))
        )

    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 2)
    model.optimize()

    temp = []
    for j in range(N):
        alg_obj = 0
        for k in data.index:
            alg_obj += (c[k][j] * x[k].x)
        temp.append(alg_obj)

    return max(temp)


def inner_heuristic(s, t, nodes, data, selected_scenarios, N, start_time):
    """
    inner heuristic process
    :param s: start node
    :param t: end node
    :param nodes: the list of all nodes
    :param data:
    :param selected_scenarios: the list of selected scenarios
    :param N:
    :param start_time: start time
    :return: guarantee, alg_obj, c_lb, d, Gamma, cal_time
    """
    n = len(data)
    c = data[selected_scenarios].to_numpy()
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
        # d_model.computeIIS()
        # d_model.write('model_ilp.ilp')
    Gamma = 1


    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma) # no timelimit
    beta = cal_beta(c, c_lb, d, n, N, Gamma)

    cal_time = time.time() - start_time

    alg_obj = get_alg_value(c_lb, d, N, c, Gamma, s, t, nodes, data)

    # alg_obj = 0
    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time

def outer_heuristic(s, t, nodes, data, selected_scenarios, N, start_time):
    """
    outer heuristic process
    :param s: start node
    :param t: end node
    :param nodes: the list of all nodes
    :param data:
    :param selected_scenarios: the list of selected scenarios
    :param N:
    :param start_time: start time
    :return: guarantee, alg_obj, c_lb, d, Gamma, cal_time
    """
    n = len(data)
    c = data[selected_scenarios].to_numpy()

    # To get the outer budgeted set
    c_lb = []
    for i in range(n):
        c_lb.append(min([cj for cj in c[i]]))
    d = []
    for i in range(n):
        d.append(max([(cj - c_lb[i]) for cj in c[i]]))
    Gamma = max([sum([(c[i][j]-c_lb[i])/d[i] for i in range(n) if d[i] != 0]) for j in range(N)])


    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma)
    beta = cal_beta(c, c_lb, d, n, N, Gamma)
    cal_time = time.time() - start_time

    alg_obj = get_alg_value(c_lb, d, N, c, Gamma, s, t, nodes, data)


    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time

def random_generate_clb_d(n, c):
    """
    randomly generate a budgeted set
    :param n:
    :param c:
    :return:
    """
    cmax = np.max(c)
    cmin = np.min(c)
    c_lb = np.random.uniform(0.8*cmin, 1.2*cmax, size=n)
    d = np.random.uniform(0, 1.2 * (cmax - cmin), size=n)
    return c_lb, d

def get_around_solution(c, c_lb, d, n, Gamma):
    """
    Generate a new solution set in the neighborhood
    :param c:
    :param c_lb:
    :param d:
    :param n:
    :param Gamma:
    :return: c_lb, d, Gamma
    """
    cmax = np.max(c)
    cmin = np.min(c)
    noise_strength = cmax - cmin
    c_lb = c_lb + np.clip(np.random.normal(0, noise_strength, size=c_lb.shape), -c_lb, np.inf)
    d = d + np.clip(np.random.normal(0, noise_strength, size=d.shape), -d, np.inf)
    Gamma = Gamma + np.clip(np.random.normal(0, noise_strength), -Gamma, n - Gamma)
    return c_lb, d, Gamma

def sampling_heuristic(s, t, nodes, data, selected_scenarios, N, start):
    """
    sampling heuristic process
    :param s: start node
    :param t: end node
    :param nodes: the list of all nodes
    :param data:
    :param selected_scenarios: the list of selected scenarios
    :param N:
    :param start_time: start time of whole process
    :return: guarantee, alg_obj, c_lb, d, Gamma, cal_time
    """
    n = len(data)
    c = data[selected_scenarios].to_numpy()
    Gamma = 1

    guarantee_list = []
    alpha_list = []
    beta_list = []
    Gamma_best = Gamma

    # randomly generate initial solution
    c_lb, d = random_generate_clb_d(n, c)
    delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma)
    beta = cal_beta(c, c_lb, d, n, N, Gamma)

    guarantee_list.append(alpha * beta)
    alpha_list.append(alpha)
    beta_list.append(beta)
    c_lb_best = c_lb
    d_best = d


    start_time = time.time()
    while time.time() - start_time < 180:
        c_lb, d, Gamma = get_around_solution(c, c_lb_best, d_best, n, Gamma_best) # Generate a new solution in neighborhood of the best solution
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
    alg_obj = get_alg_value(c_lb, d, N, c, Gamma, s, t, nodes, data)

    return guarantee_list[-1], alg_obj, c_lb_best.tolist(), d_best.tolist(), Gamma_best, cal_time



def heuristic_algs(inp):
    """
    For 10 pairs of fixed s-t, three algorithms are executed respectively
    :param inp (list): [data, nodes, selected_scenarios, N]
    :return:
    """
    data, nodes, selected_scenarios, N = inp
    selected_pairs = [[398,518], [144,41], [407,516], [257,341], [65,437], [343,481], [352,340], [442,474], [124,436], [306,344]]

    res = {'sampling_guarantee': [], 'inner_guarantee': [], 'outer_guarantee': [],
           'sampling_cal_time': [],  'inner_cal_time': [], 'outer_cal_time': [],
            'N_list': [], 'n_list': [], 'start_node': [i[0] for i in selected_pairs], 'end_node': [i[1] for i in selected_pairs],
           'inner_actual': [], 'outer_actual': [], 'sampling_actual': [],
           'sampling_gamma': [], 'inner_gamma': [], 'outer_gamma': [],
           'inner_c_lb': [], 'outer_c_lb': [], 'sampling_c_lb': [],
           'inner_d': [], 'outer_d': [], 'sampling_d': [],}

    for s, t in selected_pairs:
        opt_obj, lines = shortest_path_discrete(data, nodes, s, t, selected_scenarios)
        res['N_list'].append(N)

        # inner
        start = time.time()
        try:
            guarantee1, alg_obj1, c_lb1, d1, gamma1, cal_time1 = inner_heuristic(s, t, nodes, data, selected_scenarios, N, start)
            res['inner_actual'].append(alg_obj1 / opt_obj)
            res['inner_guarantee'].append(guarantee1)
            res['inner_gamma'].append(gamma1)
            res['inner_d'].append(d1)
            res['inner_c_lb'].append(c_lb1)
            res['inner_cal_time'].append(cal_time1)
        except Exception as e:
            res['inner_actual'].append(-1)
            res['inner_guarantee'].append(-1)
            res['inner_gamma'].append(-1)
            res['inner_d'].append(-1)
            res['inner_c_lb'].append(-1)
            res['inner_cal_time'].append(time.time() - start)


        # outer
        start = time.time()
        try:
            guarantee2, alg_obj2, c_lb2, d2, gamma2, cal_time2 = outer_heuristic(s, t, nodes, data, selected_scenarios, N, start)
            res['outer_actual'].append(alg_obj2 / opt_obj)
            res['outer_guarantee'].append(guarantee2)
            res['outer_gamma'].append(gamma2)
            res['outer_d'].append(d2)
            res['outer_c_lb'].append(c_lb2)
            res['outer_cal_time'].append(cal_time2)
        except Exception as e:
            res['outer_actual'].append(-1)
            res['outer_guarantee'].append(-1)
            res['outer_gamma'].append(-1)
            res['outer_d'].append(-1)
            res['outer_c_lb'].append(-1)
            res['outer_cal_time'].append(time.time() - start)

        # sampling
        start = time.time()
        try:
            guarantee3, alg_obj3, c_lb3, d3, gamma3, cal_time3 = sampling_heuristic(s, t, nodes, data, selected_scenarios, N, start)
            res['sampling_actual'].append(alg_obj3 / opt_obj)
            res['sampling_guarantee'].append(guarantee3)
            res['sampling_gamma'].append(gamma3)
            res['sampling_d'].append(d3)
            res['sampling_c_lb'].append(c_lb3)
            res['sampling_cal_time'].append(cal_time3)
        except Exception as e:
            res['sampling_actual'].append(-1)
            res['sampling_guarantee'].append(-1)
            res['sampling_gamma'].append(-1)
            res['sampling_d'].append(-1)
            res['sampling_c_lb'].append(-1)
            res['sampling_cal_time'].append(time.time() - start)

        # save result
        with open('./res_exp8_' + str(N), 'w') as f:
            json.dump(res, f)

def read_res_8(path):
    res = {}
    for j in range(50, 3001, 50):
        try:
            with open(path+str(j), 'r', encoding='utf-8') as file:
                temp = json.load(file)
        except:
            continue
        for k in temp:
            if k not in res:
                res[k] = temp[k]
            else:
                res[k] = res[k]+temp[k]
    return res


def to_decimal_label(y, pos):
    val = 10**y
    if val >= 1:
        return f'{val:.0f}'
    else:
        return f'{val:g}'

def show_only_powers_of_10(y, pos):
    val = 10**y
    if np.isclose(np.log10(val) % 1, 0, atol=1e-6):
        if val <= 1:
            return f"{val:.2g}"
        else:
            return f"{int(val):d}"
    else:
        return ""

def plot_res8(res, y, x_label, y_label, if_log, classes, out_path, other=''):

    plt.figure()
    for c in classes:
        if c == 'sampling':
            co = 'red'
        elif c == 'iterative':
            co = 'blue'
        elif c == 'inner':
            co = 'pink'
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
            plt.plot(y, np.log10([sum(res[c + '_'+ y_label][
                             res[x_label + '_list'].index(j):res[x_label + '_list'].index(j) + res[
                                 x_label + '_list'].count(j)]) / res[x_label + '_list'].count(j) if j in res[
                x_label + '_list'] else 0 for j in y]), color=co, label=lb, linestyle=ls)

        else:
            plt.plot(y, [sum(res[c + '_'+ y_label][
                             res[x_label + '_list'].index(j):res[x_label + '_list'].index(j) + res[
                                 x_label + '_list'].count(j)]) / res[x_label + '_list'].count(j) if j in res[
                x_label + '_list'] else 0 for j in y], color=co, label=lb, linestyle=ls)

    plt.xlabel(x_label)

    if if_log == 1:
        if y_label == 'cal_time':
            plt.ylabel('runtime(s)')
        else:
            plt.ylabel(y_label)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(to_decimal_label))
        l = 'log'
    else:
        if y_label == 'gamma':
            plt.ylabel(r'$\Gamma$')
        else:
            plt.ylabel(y_label)
        l = ''
    plt.legend()

    plt.savefig(out_path+y_label+'_'+other+l+'.pdf', bbox_inches='tight', dpi=300)

def plot_res81(res, y, x_label, y_label, if_log, classes, out_path, other=''):

    fig, ax = plt.subplots(figsize=(6,4))
    for c in classes:
        if c == 'sampling':
            co = 'red'
        elif c == 'iterative':
            co = 'blue'
        elif c == 'inner':
            co = 'pink'
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

            ax.plot(y, np.log10([sum(res[c + '_'+ y_label][
                             res[x_label + '_list'].index(j):res[x_label + '_list'].index(j) + res[
                                 x_label + '_list'].count(j)]) / res[x_label + '_list'].count(j) if j in res[
                x_label + '_list'] else 0 for j in y]), color=co, label=lb, linestyle=ls)

        else:
            plt.plot(y, [sum(res[c + '_'+ y_label][
                             res[x_label + '_list'].index(j):res[x_label + '_list'].index(j) + res[
                                 x_label + '_list'].count(j)]) / res[x_label + '_list'].count(j) if j in res[
                x_label + '_list'] else 0 for j in y], color=co, label=lb, linestyle=ls)

    ax.set_xlabel(x_label)
    if if_log == 1:
        if y_label == 'guarantee':
            ax.set_ylabel('guarantee')
        elif y_label == 'cal_time':
            ax.set_ylabel('runtime(s)')
        else:
            ax.set_ylabel(y_label)

        temp = [i for i in range(0, 10001, 1000)]
        # temp.append(0.1)
        temp.append(1)
        temp.append(10)
        temp.append(100)
        yticks_original = np.array(temp)
        ax.set_yticks(np.log10(np.clip(yticks_original, 1, None)))
        ax.yaxis.set_major_formatter(FuncFormatter(show_only_powers_of_10))
        ax.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        l = 'log'
    else:
        plt.ylabel(y_label)
        l = ''
    # plt.legend()

    plt.savefig(out_path+y_label+'_'+other+l+'.pdf', bbox_inches='tight', dpi=300)



def plot_results(dir=None):
    if dir == None:
        path = './res_exp8_'
        out_path = './'
    else:
        path = './'+ dir +'/res_exp8_'
        out_path = './'+dir+'/'
    res8 = read_res_8(path)
    for y_label in ['guarantee', 'gamma', 'cal_time', 'actual']:
        plot_res8(res8, range(50, 1001, 50), 'N', y_label, 0, ['sampling', 'inner', 'outer'], out_path, other='')
        plot_res81(res8, range(50, 1001, 50), 'N', y_label, 1, ['sampling', 'inner', 'outer'], out_path, other='')

if __name__ == '__main__':
    data = pd.read_csv('./Chicago_main.csv')
    nodes = sorted(set(data["start_node"]).union(set(data["end_node"])))
    scenario_cols = [col for col in data.columns[8:]]
    random.seed(42)
    random.shuffle(scenario_cols)

    # for an example
    # N = 50
    # selected_scenarios = scenario_cols[: N]
    # heuristic_algs([data, nodes, selected_scenarios, N])

    # for multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    N_list = [[data[['start_node','end_node']+scenario_cols[:i]], nodes, scenario_cols[:i], i] for i in range(50, 3001, 50)]
    with Pool(processes=40) as pool:
        result = pool.map(heuristic_algs, N_list)

    plot_results()





