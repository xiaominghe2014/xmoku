from tic_tac_toe import TicTacToe
import numpy as np
from net import CNNModel
import torch
import numpy as np
import os
# 加载模型
# 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel()

model_path = 'model/model.tar'

if os.path.exists(model_path):
    model_cfg = torch.load(model_path)
    model.load_state_dict(model_cfg['model'])

# 将模型设置为评估模式
model.eval()

"""
训练说明:
如果已经有模型的话：
    1. 可以调用模型训练
    2. 如果有效操作全部为唯一结果，则本局训练结束，录入数据，可以迭代训练，向前回溯胜率计算
"""

NUM_GAMES = 1_000
train_data = []
train_labels = []

def prior_call(g):
    board = g.board
    data_tensor = torch.Tensor(board).unsqueeze(0).unsqueeze(0)
    # print(f'data_tensor: {data_tensor}')
    with torch.no_grad():
        output, loss = model(data_tensor,None)
        # output = model(data_tensor)
        output = torch.softmax(output, dim=1)
        output = output.detach().numpy().tolist()[0]
        # 将output前三个元素各加 output 最后一个元素的1/3
        output[:3] = [x + output[-1]/3 for x in output[:3]]
        output[:3] = [x/sum(output[:3]) for x in output[:3]]
    return output

for i in range(NUM_GAMES):
    game = TicTacToe(prior_call)
    while game.winner is None:
       # 获取当前玩家的下一步动作
        action = game.get_best_action()
        if not action:
            print(f'无可操作的步骤')
            break
        # 在游戏中落子
        [x,y]=action
        r = game.make_move(x, y)
        if not r:
            raise Exception('发生了错误')
        print(f'落子：{game.cell_char_at(x, y)}->({x},{y})')
        game.print_board()
        game.check_win()
        if game.winner is not None:
            print(f'第{i+1}局游戏 {game.winner} 获胜')

    state = game.get_train_data()
    train_data.append(state)

# train_data = np.array(train_data)
# train_labels = np.array(train_labels)

# np.save('game_data/train_data.npy', train_data)
# np.save('game_data/train_labels.npy', train_labels)

torch.save({
    'data':train_data,
},'game_data/train_data.tar')
