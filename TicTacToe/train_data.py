from ticTacToe import TicTacToe
import numpy as np

"""
训练说明:
如果已经有模型的话：
    1. 可以调用模型训练
    2. 如果有效操作全部为唯一结果，则本局训练结束，录入数据，可以迭代训练，向前回溯胜率计算
"""

NUM_GAMES = 1_000
train_data = []
train_labels = []

for i in range(NUM_GAMES):
    game = TicTacToe()
    while game.winner is None:
       # 获取当前玩家的下一步动作
        action = game.get_best_action()
        if not action:
            print(f'无可操作的步骤')
            break
        # 在游戏中落子
        [x,y]=action
        print(f'落子：({x},{y})')
        r = game.make_move(x, y)
        if not r:
            raise Exception('发生了错误')
        game.print_board()
        game.check_win()
        if game.winner is not None:
            print(f'第{i}局游戏 {game.winner} 获胜')
    state = game.get_train_data()
    train_data.append(state[0])
    train_labels.append(state[1])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

np.save('game_data/train_data.npy', train_data)
np.save('game_data/train_labels.npy', train_labels)
