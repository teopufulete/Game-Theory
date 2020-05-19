import numpy as np
import scipy.optimize

class ZeroSumGame:
    def __init__(self, get_score, action_1, action_2):
        self._get_score = get_score     #returns game score
        self._action_1 = action_1       #player 1s number of possible actions
        self._action_2 = action_2       #player 2s number of possible actions
        self.nash_eq = None

    def player_1_best_strategy(self):
        i = len(self._action_1)
        j = len(self._action_2)

        cost_matrix = np.array(
            [[self._get_score(act1, act2) for act1 in self._action_1]
             for act2 in self._action_2])

        c = np.zeros(n + 1)     #n = probabilities of actions for player 1
        c[n] = -1

        #minimizing loss: in worst case outcome player 1 gets at least the cost
        A_ub = np.hstack(-cost_matrix, np.ones((j, 1)))
        b_ub = np.zeros((j))

        #normalizing probabilities
        A_eq = np.ones((1, i + 1))
        A_eq[0][i] = 0
        b_eq = np.ones(1)

        #probabilities can't be negative, cost can
        bounds = [(0, None) for _ in range(n)] + [(None, None)]

        resolve = scipy.optimize.linprog(c = c, A_ub = A_ub, b_ub  = b_ub,
                                     A_eq = A_eq, b_eq = b_eq,
                                     bounds=bounds)


        #equilibriumm is a distribution over player 1's actions
        self.nash_eq = resolve.x[:i]
        return -resolve.fun

    def generate_action(self):
        return np.random.choice(self._first_actions, p = self.optimal_policy)
