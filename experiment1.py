"""
Experiment 1:
    implement inner/outer heuristic
    test the relationship between the approximation guarantee
"""

from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import scipy
import scipy.stats
import sklearn.datasets
import time


def get_opt_value(c, n, N, P):
    """
    calculate the nominal problem to get the optimal value
    """
    model = Model()
    z = model.addVar(vtype=GRB.CONTINUOUS, name='z')
    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    model.setObjective(z, GRB.MINIMIZE)
    model.addConstrs(z >= quicksum(c[i][j] * x[i] for i in range(n)) for j in range(N))
    model.addConstr(quicksum(x[i] for i in range(n)) == P)
    model.optimize()
    return model.ObjVal


def get_alg_value(c_lb, d, n, N, c, Gamma, P):
    """
    calculate the algorithm value
    :param c_lb (list): c bar of budgeted data set
    :param d (list): d of budgeted data set
    :param n (int):
    :param N (int):
    :param c (list[list]): discrete data set
    :param Gamma (float):
    :param P:
    :return: the algorithm value (int)
    """
    model = Model()
    x = model.addVars(n, vtype=GRB.BINARY, name='x')
    pi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='pi')
    p = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name='p')
    model.setObjective(quicksum(c_lb[i] * x[i] for i in range(n)) + Gamma * pi + quicksum(p[i] for i in range(n)))
    model.addConstrs(pi + p[i] >= d[i] * x[i] for i in range(n))
    model.addConstr(quicksum(x[i] for i in range(n)) == P)
    model.optimize()

    temp = []
    for j in range(N):
        alg_obj = 0
        for i in range(n):
            alg_obj += (c[i][j] * x[i].x)
        temp.append(alg_obj)

    return max(temp)


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



def inner_heuristic(c, n, N, P, start_time):
    """
    inner heuristic algorithm
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

    alg_obj = get_alg_value(c_lb, d, n, N, c, Gamma, P)

    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time


def outer_heuristic(c, n, N, P, start_time):
    """
    outer heuristic algorithm
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
    alg_obj = get_alg_value(c_lb, d, n, N, c, Gamma, P)

    return alpha*beta, alg_obj, c_lb, d, Gamma, cal_time

def cal_correlation(x, y):
    """
    calculate the correlation between x and y
    :param x:
    :param y:
    :return:
    """
    s1 = pd.Series(x)
    s2 = pd.Series(y)

    return s1.corr(s2)

def main(a):
    SSIZE = a
    n = 10
    N = 50

    cor_list = []
    for P in [5]:
        for idx in [6]:#range(1, 7):  # data generation method index, each index represent a way
            index = str(idx)
            outer = [] # guarantees list
            outer_actual = [] # actual list
            inner = []
            inner_actual = []
            outer_ins = []
            outer_ins_actual = []
            np.random.seed(50)
            for t in range(500):
                c = []
                if idx == 1:
                    for i in range(n):
                        temp = np.random.uniform(5, 210, size=N)
                        c.append(temp)
                if idx == 2:
                    for i in range(n):
                        ub = np.random.uniform(5, 210)
                        lb = np.random.uniform(5, ub)
                        temp = np.random.uniform(lb, ub, size=N)
                        c.append(temp)

                if idx == 3:
                    for i in range(n):
                        ub = np.random.uniform(5 + i * 20, 5 + (i + 1) * 20)
                        lb = np.random.uniform(5 + i * 20, ub)
                        temp = np.random.uniform(lb, ub, size=N)
                        c.append(temp)

                if idx == 4:
                    for i in range(n):
                        lb = 5 + i * 20
                        ub = 5 + (i + 1) * 20
                        temp = np.random.uniform(lb, ub, size=N)
                        c.append(temp)
                if idx == 5:
                    for i in range(n):
                        a, b = 5, 210
                        mu, sigma = 100, 50
                        trunc_normal = truncnorm(a=(a - mu) / sigma, b=(b - mu) / sigma, loc=mu, scale=sigma)
                        temp = trunc_normal.rvs(N)
                        c.append(temp)
                if idx == 6:
                    Sigma = SSIZE * sklearn.datasets.make_spd_matrix(n_dim=n)
                    mean = []
                    temp = []
                    for j in range(n):
                        mean.append((25 + 100 * scipy.stats.uniform.rvs()))
                    for i in range(N):
                        x = scipy.stats.multivariate_normal.rvs(mean=mean, cov=Sigma)
                        temp.append(x)
                    for i in range(N):
                        tmp = []
                        for j in range(n):
                            tmp.append(max(0, temp[i][j]+300))
                        c.append(tmp)
                    c = np.array(c).T

                # get the optimal value of nominal problem
                opt_obj = get_opt_value(c, n, N, P)

                 # outer
                start = time.time()
                guarantee1, alg_obj1, c_lb1, d1, Gamma1, cal_time1 = outer_heuristic(c, n, N, P, start)
                outer.append(guarantee1)
                outer_actual.append(alg_obj1 / opt_obj)

                # inner
                # start = time.time()
                # guarantee2, alg_obj2, c_lb2, d2, Gamma2, cal_time2 = inner_heuristic(c, n, N, P, start)
                # inner.append(guarantee2)
                # inner_actual.append(alg_obj2 / opt_obj)
                #
                # # random
                # c_lb = np.random.uniform(5,210, size=n)
                # d = np.random.uniform(5,210, size=n)
                # guarantee2, alg_obj2 = outer_instance(c, c_lb, d, n, N, P)
                # outer_ins.append(guarantee2)
                # outer_ins_actual.append(alg_obj2 / opt_obj)


            # calculate correlation
            cor1 = cal_correlation(outer_actual, outer)
            print("Correlation: ", cor1)

            # plot results
            plt.figure()
            if idx in [3, 4]:
                plt.xlim(0.8, 1.2)
                plt.xticks([0.8, 1, 1.2])
            plt.scatter(outer_actual, outer, color='black', marker='o', s=5, label='outer', alpha=0.6)
            plt.xlabel('actual')
            plt.ylabel('guarantee')
            # plt.legend()
            # plt.show()
            plt.savefig('./heuristic_outer_' + str(P) + '_' + str(SSIZE) + '_' + index + '.pdf', bbox_inches='tight', dpi=300)

    return

if __name__ == '__main__':
    main(80) # SSIZE=80, ploting experiment 1 result
