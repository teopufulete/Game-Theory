import numpy as np
import scipy.optimize


class ZeroSumGame:
    def __init__(self, get_score, action_1, action_2):
        self._get_score = get_score     # returns game score
        self._actions = [action_1, action_2]
        self.nash_eq = None
        self.game_cost = self.get_optimal_strategy()
        

    def get_optimal_policy(self):
        x = len(self._actions[0])
        y = len(self._actions[1])

        cost_matrix = np.array(
            [[self._get_score(act_1, act_2) for act_2 in self._actions[1]]
             for act_1 in self._actions[0]]).transpose((2, 0, 1))

        total_cost = cost_matrix[0] + cost_matrix[1]
        vars_count = x + y + 2

        def loss(i):
            first, second = i[:x], i[:x, x + y]
            alpha,beta = i[x + y], i[x + y + 1]
            return -first @ total_cost @ second + alpha + beta

        def jac(i):
            first, second = i[:x], i[x:x + y]
            first_jac = -total_cost @ second
            second_jac = -first @ total_cost
            alpha_jac = [1]
            betta_jac = [1]
            return np.concatenate((first_jac, second_jac,
                                   alpha_jac, betta_jac))
                                   
       # Gx >= 0
        G = np.zeros((x + y, vars_count))
        for j in range(x):
            G[j][x:x + y] = -cost_matrix[0][j]
            G[j][x + y] = 1
        for j in range(y):
            G[x + j][:x] = -cost_matrix[1][:, j]
            G[x + j][x + y + 1] = 1

        # Ax - b = 0
        A = np.zeros((2, vars_count))
        A[0][:x] = 1
        A[1][x:x + y] = 1
        b = np.ones(2)

        constraints = [
            {   'type': 'ineq',
                'fun': lambda x: G @ x,
                'jac': lambda x: G,},
            {   'type': 'eq',
                'fun': lambda x: A @ x - b,
                'jac': lambda x: A}]

        bounds = [(0, None) for _ in range(x + y)]
        bounds += [(None, None), (None, None)]

        x0 = np.random.random(vars_count)
        x0[:x] /= x0[:x].sum()
        x0[x:x + y] /= x0[x:x + y].sum()
        x0[x + y:x + y + 2] *= total_cost.sum()

        resolve = scipy.optimize.minimize(x0 = x0, fun = loss, jac = jac, method = 'SLSQP',
                                      constraints = constraints,
                                      bounds = bounds)
                         
                         
    def gen_action(self, player):
        assert player in {1, 2}
        player -= 1
        return np.random.choice(self._actions[player], p = self.nash_eq[player])
