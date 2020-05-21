import numpy as np
import scipy.optimize


class NonZeroSumGame:
    def __init__(self, get_score, action_1, action_2):
        self._get_score = get_score
        self._actions = [action_1, action_2]
        self.nash_eq = None
        self.game_cost = self.get_optimal_strategy()


    def get_optimal_strategy(self):
        n = len(self._actions[0])
        m = len(self._actions[1])

        cost_matrix = np.array(
            [[self._get_score(act_1, act_2) for act_2 in self._actions[1]]
             for act_1 in self._actions[0]]).transpose((2, 0, 1))

        total_cost = cost_matrix[0] + cost_matrix[1]
        vars_count = n + m + 2


        def loss(x):
            first, second = x[:n], x[n:n + m]
            alpha, betta = x[n + m], x[n + m + 1]
            return -first @ total_cost @ second + alpha + betta


        def jac(x):
            first, second = x[:n], x[n:n + m]
            first_jac = -total_cost @ second
            second_jac = -first @ total_cost
            alpha_jac = [1]
            betta_jac = [1]
            return np.concatenate((first_jac, second_jac,
                                   alpha_jac, betta_jac))


        # Gx >= 0
        G = np.zeros((n + m, vars_count))
        for i in range(n):
            G[i][n:n + m] = -cost_matrix[0][i]
            G[i][n + m] = 1
        for i in range(m):
            G[n + i][:n] = -cost_matrix[1][:, i]
            G[n + i][n + m + 1] = 1


        # Ax - b = 0
        A = np.zeros((2, vars_count))
        A[0][:n] = 1
        A[1][n:n + m] = 1
        b = np.ones(2)

        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: G @ x,
                'jac': lambda x: G,
            },
            {
                'type': 'eq',
                'fun': lambda x: A @ x - b,
                'jac': lambda x: A
            }
        ]

        bounds = [(0, None) for _ in range(n + m)]
        bounds += [(None, None), (None, None)]

        x0 = np.random.random(vars_count)
        x0[:n] /= x0[:n].sum()
        x0[n:n + m] /= x0[n:n + m].sum()
        x0[n + m:n + m + 2] *= total_cost.sum()

        resolve = scipy.optimize.minimize(x0 = x0, fun = loss, jac = jac, method = 'SLSQP',
                                      constraints = constraints,
                                      bounds = bounds)


        self.nash_eq = resolve.x[:n], resolve.x[n:n + m]
        game_cost = [self.nash_eq[0] @ cost_matrix[player] @ self.nash_eq[1]
                      for player in (0, 1)]
        return game_cost


    def generate_action(self, player):
        assert player in {1, 2}
        player -= 1
        return np.random.choice(self._actions[player], p = self.nash_eq[player])
