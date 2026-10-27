"""
Experiment 2:
    Compare the performance of the three heuristics IH, OH, and SH to the exact algorithm IE.
"""

from gurobipy import *
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import json
from multiprocessing import Pool
import scipy
import sklearn



def random_generate_clb_d(n):
    """
    randomly generate a budgeted set
    :param n:
    :param c:
    :return:
    """
    c_lb = np.random.uniform(5, 210, size=n)
    d = np.random.uniform(5, 210, size=n)
    return c_lb, d


def cal_alpha(c, c_lb, d, n, N, Gamma, start_time=0, if_exact=False):
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
    if if_exact:
        model.setParam("TimeLimit", max(10, 7200 - (time.time() - start_time)))
    elif start_time > 0:
        model.setParam("TimeLimit", max(5, 180-(time.time()-start_time)))
    else:
        model.setParam("TimeLimit", 300)
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


def get_around_solution(c_lb, d, n, Gamma):
    """
    Generate a new solution set in the neighborhood
    :param c_lb:
    :param d:
    :param n:
    :param Gamma:
    :return: c_lb, d, Gamma
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
    is_valid = False
    while is_valid == False:
        c_lb, d = random_generate_clb_d(n)
        delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start)
        beta = cal_beta(c, c_lb, d, n, N, Gamma, start)
        init_guarantee = alpha * beta
        if init_guarantee >= 1:
            guarantee_list.append(alpha * beta)
            alpha_list.append(alpha)
            beta_list.append(beta)
            c_lb_best = c_lb
            d_best = d
            is_valid = True


    # sampling
    while time.time() - start < 180:
        c_lb, d, Gamma = get_around_solution(c_lb_best, d_best, n, Gamma_best) # Generate a new solution in neighborhood of the best solution
        try:
            delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start)
            beta = cal_beta(c, c_lb, d, n, N, Gamma, start)
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
    alg_obj = get_alg_value(c_lb_best, d_best, n, N, c, Gamma_best)

    return guarantee_list[-1], Gamma_best, c_lb_best, d_best, alg_obj, cal_time

def get_obj_lb(c, delta_list, n, N, start_time):
    """
    Gamma variable
    :param c:
    :param delta_list:
    :param n:
    :param N:
    :return:
    """
    K = len(delta_list)
    model = Model()
    e = 0.001

    # variables
    t = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='t')
    c_lb = model.addVars(n, vtype=GRB.CONTINUOUS, name='c_lb', lb=0)
    d = model.addVars(n, vtype=GRB.CONTINUOUS, name='d', lb=0)  # d
    Gamma = model.addVar(vtype=GRB.CONTINUOUS, name='Gamma', lb=1, ub=N)
    zdelta = model.addVars(n, N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='zdelta')
    lam = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='lam')
    z = model.addVars(K, vtype=GRB.BINARY, name='z')
    for i in range(n):
        for k in range(K):
            model.addGenConstrIndicator(z[k], 0, c_lb[i] + d[i] * delta_list[k][i] - quicksum(lam[k, j] * c[i][j] for j in range(N)), GRB.LESS_EQUAL, 0)
    # model.addConstrs(c_lb[i] + d[i] * delta_list[k][i] <= quicksum(lam[k, j] * c[i][j] for j in range(N)) + M * z[k] for i in range(n) for k in range(K))

    model.addConstrs(quicksum(delta_list[k][i] for i in range(n)) >= z[k] * Gamma + e for k in range(K))
    for k in range(K):
        model.addGenConstrIndicator(z[k], 0, quicksum(delta_list[k][i] for i in range(n)) - Gamma, GRB.LESS_EQUAL, 0)
    # model.addConstrs(quicksum(delta_list[k][i] for i in range(n)) <= Gamma + z[k] * M for k in range(K))

    model.addConstrs(quicksum(lam[k, j] for j in range(N)) == 1 for k in range(K))
    model.addConstrs(t * c[i][j] <= c_lb[i] + d[i] * zdelta[i, j] for i in range(n) for j in range(N))
    model.addConstrs(quicksum(zdelta[i, j] for i in range(n)) <= Gamma for j in range(N))
    model.setObjective(t, GRB.MAXIMIZE)
    model.setParam("TimeLimit", max(10, 7200 - (time.time() - start_time)))
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 2)
    model.optimize()

    # if model.status == GRB.INFEASIBLE:
    # model.computeIIS()
    # model.write('model_ilp.ilp')

    return [c_lb[i].x for i in range(n)], [d[i].x for i in range(n)], 1 / model.ObjVal, Gamma.x



def iterative(c, n, N):
    """
    iterative exact algorithm
    1. set alpha=1, beta=1, solve main problem, get c_lb
    2. back to alpha problem, to find a new worse case delta, add it to delta list
    3. repeat, every time have one more delta
    4. sub problem's obj as lower bound, alpha problem's obj as upper bound
    without timelimit
    Gamma is variable
    """

    delta_list = [[1/n for i in range(n)]]
    lb_list = []
    ub_list = []
    start_time = time.time()
    Gamma_best = 1

    while time.time() - start_time <= 7200:  # avoid dead loop
        try:
            c_lb, d, obj_lb, Gamma = get_obj_lb(c, delta_list, n, N, start_time)
            Gamma_best = Gamma
            delta, alpha = cal_alpha(c, c_lb, d, n, N, Gamma, start_time, if_exact=True)
            delta_list.append(delta)
            lb_list.append(obj_lb)
            ub_list.append(alpha * obj_lb)
            if alpha <= 1.01:
                print(alpha, 'alpha <= 1 break')
                break
        except:
            alpha= -1
            c_lb = -1
            d = -1
            break

    return lb_list, ub_list, alpha, Gamma_best, c_lb, d



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
    outer heuristic
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

def plot_results(dir=None):
    if dir == None:
        path = './res_exp2_'
        out_path = './'
    else:
        path = './' + dir + '/res_exp2_'
        out_path = './' + dir + '/'

    res2 = {}
    for j in range(2, 41):
        try:
            with open(path + str(j), 'r', encoding='utf-8') as file:
                temp = json.load(file)
                temp['N_list'] = [j for t in range(len(temp['if_optimal']))]
                if len(temp['if_optimal']) != 10:
                    print(j)
        except:
            temp = {}

        for k in temp:
            if k not in res2:
                res2[k] = temp[k]
            else:
                res2[k] = res2[k] + temp[k]

    N_list = sorted(set(res2['N_list']))
    plt.figure()
    plt.plot(N_list, [
        sum(res2['iterative_guarantee'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) /
        res2['N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='IE',
             linestyle=':')
    plt.plot(N_list, [
        sum(res2['sampling_guarantee'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) /
        res2['N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='SH',
             linestyle='-.')
    plt.plot(N_list, [
        sum(res2['inner_guarantee'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='IH',
             linestyle='--')
    plt.plot(N_list, [
        sum(res2['outer_guarantee'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='OH', linestyle='-')

    plt.xlabel('N')
    plt.ylabel('guarantee')
    # plt.title('')
    plt.legend()
    # plt.show()
    plt.savefig(out_path+'res2_guarantee.pdf', bbox_inches='tight', dpi=300)
    print()
    #
    plt.figure()
    # plt.plot(N_list, [
    #     sum(res2['iterative_actual'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) /
    #     res2['N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='IE',
    #          linestyle=':')
    plt.plot(N_list, [
        sum(res2['sampling_actual'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) /
        res2['N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='SH',
             linestyle='-.')
    plt.plot(N_list, [
        sum(res2['inner_actual'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='IH',
             linestyle='--')
    plt.plot(N_list, [
        sum(res2['outer_actual'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='OH', linestyle='-')

    plt.xlabel('N')
    plt.ylabel('guarantee')
    # plt.title('')
    plt.legend()
    # plt.show()
    plt.savefig(out_path+'res2_actual.pdf', bbox_inches='tight', dpi=300)
    print()
    #
    plt.figure()
    plt.plot(N_list, [
        sum(res2['iterative_gamma'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='IE', linestyle=':')
    plt.plot(N_list, [
        sum(res2['sampling_gamma'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='SH',
             linestyle='-.')
    # plt.plot(N_list, [sum(res2['inner_gamma'][res2['N_list'].index(j):res2['N_list'].index(j)+res2['N_list'].count(j)]) / res2['N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='pink', label='inner,n=3')
    plt.plot(N_list, [
        sum(res2['outer_gamma'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
            'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list], color='black', label='OH', linestyle='-')
    plt.xlabel('N')
    plt.ylabel(r'$\Gamma$')
    # plt.title('')
    plt.legend()
    plt.savefig(out_path+'res2_gamma.pdf', bbox_inches='tight', dpi=300)

    # log
    def to_decimal_label(y, pos):
        val = 10 ** y
        if val >= 1:
            return f'{val:.0f}'  # >=1 时显示整数
        else:
            return f'{val:g}'

    def label_as_original(y, pos):
        val = 10 ** y
        if val >= 1:
            return f"{val:.0f}"
        else:
            return f"{val:g}"

    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_list, np.log10([sum(
        res2['iterative_cal_time'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
                                  'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list]), color='black',
            label='IE', linestyle=':')
    ax.plot(N_list, np.log10([sum(
        res2['sampling_cal_time'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
                                  'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list]), color='black',
            label='SH', linestyle='-.')
    ax.plot(N_list, np.log10([sum(
        res2['inner_cal_time'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
                                  'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list]), color='black',
            label='IH', linestyle='--')
    ax.plot(N_list, np.log10([sum(
        res2['outer_cal_time'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) / res2[
                                  'N_list'].count(j) if j in res2['N_list'] else 0 for j in N_list]), color='black',
            label='OH', linestyle='-')
    ax.set_xlabel('N')
    ax.set_ylabel('runtime(s)')

    temp = [i for i in range(0, 10001, 1000)]
    temp.append(0.1)
    temp.append(1)
    temp.append(10)
    temp.append(100)
    yticks_original = np.array(temp)
    ax.set_yticks(np.log10(np.clip(yticks_original, 0.01, None)))  # 注意跳过 0，log(0)无定义

    def show_only_powers_of_10(y, pos):
        val = 10 ** y
        if np.isclose(np.log10(val) % 1, 0, atol=1e-6):
            if val <= 1:
                return f"{val:.2g}"
            else:
                return f"{int(val):d}"
        else:
            return ""

    ax.yaxis.set_major_formatter(FuncFormatter(show_only_powers_of_10))
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_path+'res2_runtime.pdf', bbox_inches='tight', dpi=300)

    #
    plt.figure()
    plt.plot(N_list, [
        sum(res2['if_optimal'][res2['N_list'].index(j):res2['N_list'].index(j) + res2['N_list'].count(j)]) if j in res2[
            'N_list'] else 0 for j in N_list], color='black', label='IE', linestyle=':')
    plt.xlabel('N')
    plt.ylabel('times achieving optimum (10 runs)')
    # plt.title('')
    plt.legend()
    plt.savefig(out_path+'res2_ifoptimal.pdf', bbox_inches='tight', dpi=300)
    print()


def main(inp):
    n, N, k = inp

    try:
        path = './res_exp2_' + str(k)
        with open(path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        for key in ['N_list',
                    'sampling_guarantee', 'sampling_actual', 'sampling_gamma',
                    'sampling_cal_time', 'sampling_c_lb', 'sampling_d',
                    'inner_guarantee', 'inner_actual', 'inner_gamma',
                    'inner_cal_time', 'inner_c_lb', 'inner_d',
                    'outer_guarantee', 'outer_actual', 'outer_gamma',
                    'outer_cal_time', 'outer_c_lb', 'outer_d'
                    ]:
            res[key] = []
    except:
        res = {'sampling_guarantee': [], 'iterative_guarantee': [], 'inner_guarantee': [], 'outer_guarantee': [],
               'sampling_cal_time': [], 'iterative_cal_time': [], 'inner_cal_time': [], 'outer_cal_time': [],
               'if_optimal': [], 'N_list': [], 'n_list': [],
               'sampling_actual': [], 'inner_actual': [], 'outer_actual': [],
               'sampling_gamma': [], 'iterative_gamma': [], 'outer_gamma': [],
               'sampling_c_lb':[], 'inner_c_lb': [], 'outer_c_lb':[], 'iterative_c_lb': [],
               'sampling_d':[], 'inner_d': [], 'outer_d':[], 'iterative_d': []}

    idx = 0
    for tt in range(10):  # repeat 10 times
        res['N_list'].append(N)
        res['n_list'].append(n)
        idx += 1
        np.random.seed(idx)

        # generate dataset
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

        # outer heuristic
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

        # inner heuristic
        start = time.time()
        try:
            guarantee2, alg_obj2, c_lb2, d2, Gamma2, cal_time2 = inner_heuristic(c, n, N, start)
            res['inner_guarantee'].append(guarantee2)
            res['inner_actual'].append(alg_obj2 / opt_obj)
            res['inner_c_lb'].append(c_lb2)
            res['inner_d'].append(d2)
            res['inner_cal_time'].append(cal_time2)
        except Exception as e:
            res['inner_guarantee'].append(-1)
            res['inner_actual'].append(-1)
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
            res['sampling_cal_time'].append(cal_time3)
            res['sampling_c_lb'].append(c_lb3.tolist())
            res['sampling_d'].append(d3.tolist())
        except Exception as e:
            res['sampling_guarantee'].append(-1)
            res['sampling_actual'].append(-1)
            res['sampling_gamma'].append(-1)
            res['sampling_c_lb'].append(-1)
            res['sampling_d'].append(-1)
            res['sampling_cal_time'].append(time.time() - start)

        # iterative exact algorithm
        start = time.time()
        lb_list, ub_list, alpha, Gamma_best4, c_lb, d = iterative(c, n, N)
        if len(ub_list) == 0:
            ub_list = [-1]
            alpha = -1
            Gamma_best4 = -1
        if alpha <= 1.01 and alpha >= 0:
            res['if_optimal'].append(1)
        else:
            res['if_optimal'].append(0)
        res['iterative_gamma'].append(Gamma_best4)
        res['iterative_cal_time'].append(time.time() - start)
        res['iterative_guarantee'].append(ub_list[-1])
        res['iterative_c_lb'].append(c_lb)
        res['iterative_d'].append(d)

        with open('./res_exp2_'+str(k), 'w') as f:
            json.dump(res, f)

    return

if __name__ == '__main__':
    # main([4, 5, 100])   [n, N, k=file_index]
    from multiprocessing import freeze_support
    freeze_support()
    dataset = [(3, i, i) for i in range(2, 41)] # fix n=3, vary N from [2, 40]
    with Pool(processes=4) as pool:
        results = pool.map(main, dataset)
    plot_results()






