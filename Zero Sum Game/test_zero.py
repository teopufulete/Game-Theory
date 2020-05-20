import numpy as np
from zero_sum_game import ZeroSumGame

SEED = 1337

def simulate_game(get_score, action_1, action_2 , iters=10 ** 4, comments=None):
    def reverse_score(act_1, act_2):
        return -get_score(act_2, act_1)

    first = ZeroSumGame(get_score, action_1, action_2)
    second = ZeroSumGame(reverse_score, action_2, action_1)

    score = 0
    for it in range(iters):
        a = first.generate_action()
        b = second.generate_action()
        score += get_score(a, b)

    if comments is not None:
        print(comments)
    print(f'Average score: {score / iters}')
    print(f'Expected game price: {first.game_cost}')
    print('-' * 80)


FIGHTS = {('rock', 'scissors'),
            ('paper', 'rock'),
            ('scissors', 'paper')}

ACTIONS = ('rock', 'paper', 'scissors')


def rock_paper_scissors(act_1, act_2):
    assert act_1 in ACTIONS and act_2 in ACTIONS
    if (act_1, act_2) in FIGHTS:
        return 1
    if (act_2, act_1) in FIGHTS:
        return -1
    return 0


if __name__ == '__main__':
    np.random.seed(SEED)

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper', 'scissors'],
                  comments="Fair rock paper scissors")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors'")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors' and 'paper'")
