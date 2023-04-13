from gomoku import Gomoku
import numpy as np

NUM_GAMES = 1000
train_data = []
train_labels = []

for i in range(NUM_GAMES):
    game = Gomoku()
    while True:
       # 获取当前玩家的下一步动作
        action = game.get_best_action()
        if not action:
            print(f'无可操作的步骤')
            break
        # 在游戏中落子
        [x,y]=action
        game.play(x, y)
        # 如果当前玩家获胜，游戏结束
        if game.check_win(x, y):
            break
        state = game.get_current_state()
        train_data.append(state[0])
        train_labels.append(state[1])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

np.save('game_data/train_data.npy', train_data)
np.save('game_data/train_labels.npy', train_labels)
