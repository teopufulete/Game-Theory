import numpy as np
from non_zero_sum_game import NonZeroSumGame

SEED = 1337

def simulate_game(get_score, action_1, action_2, iters=10 ** 4, comments = None, verbosity = 1):
    game = NonZeroSumGame(get_score, action_1, action_2)

    score_1, score_2 = 0, 0
    for it in range(iters):
        act_1 = game.generate_action(1)
        act_2 = game.generate_action(2)
        f_score, s_score = get_score(act_1, act_2)
        score_1 += f_score
        score_2 += s_score

    if verbosity >= 1:
        if comments is not None:
            print(comments)
        print(game.nash_eq[0])
        print(f'Game price first: {game.game_cost[0]}')
        print(f'Average score first: {score_1 / iters}')
        print()
        print(game.nash_eq[1])
        print(f'Game price second: {game.game_cost[1]}')
        print(f'Average score second: {score_2 / iters}')
        print('-' * 80)
    return game.nash_eq, game.game_cost


def rock_paper_scissors(first_act, second_act):
    FIGHTS = {('rock', 'scissors'),
                ('paper', 'rock'),
                ('scissors', 'paper')}
    ACTIONS = ('rock', 'paper', 'scissors')

    assert first_act in ACTIONS and second_act in ACTIONS
    if (first_act, second_act) in FIGHTS:
        return 1, -1
    if (second_act, first_act) in FIGHTS:
        return -1, 1
    return 0, 0


def prisoners_dilemma(first_act, second_act):
    ACTIONS = ['silent', 'betray']
    assert first_act in ACTIONS and second_act in ACTIONS
    if first_act == 'silent' and second_act == 'silent':
        return -1, -1
    if first_act == 'silent' and second_act == 'betray':
        return -3, 0
    if first_act == 'betray' and second_act == 'betray':
        return -2, -2,
    if first_act == 'betray' and second_act == 'silent':
        return 0, -3

    
def bach_or_stravinsky(first_act, second_act):
    ACTIONS = ['bach', 'stravinsky']
    assert first_act in ACTIONS and second_act in ACTIONS
    if first_act == 'bach' and second_act == 'bach':
        return 2, 1
    if first_act == 'bach' and second_act == 'stravinsky':
        return -1, -1
    if first_act == 'stravinsky' and second_act == 'stravinsky':
        return 1, 2
    if first_act == 'stravinsky' and second_act == 'bach':
        return -1, -1

if __name__ == '__main__':
    np.random.seed(SEED)
    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper', 'scissors'],
                  comments="Fair rock paper scissors with zero price")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock', 'paper'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors'")

    simulate_game(rock_paper_scissors,
                  ['rock', 'paper', 'scissors'],
                  ['rock'],
                  comments="Unfair rock paper scissors where the second player can't use 'scissors' and 'paper'")

    simulate_game(prisoners_dilemma,
                  ['silent', 'betray'],
                  ['silent', 'betray'],
                  comments="Prisoner's dilemma")

    for i in range(10):
        simulate_game(bach_or_stravinsky,
                      ['bach', 'stravinsky'],
                      ['bach', 'stravinsky'],
                      comments="Bach or  Stravinsky? #{}".format(i))
