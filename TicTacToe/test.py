
from tic_tac_toe import TicTacToe

import os

from net import CNNModel
import torch
import numpy as np
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

def analysis(board):
    data_tensor = torch.Tensor(board).unsqueeze(0).unsqueeze(0)
    # labels = torch.tensor([-1], dtype=torch.long)
    # print(f'data_tensor: {data_tensor}')
    with torch.no_grad():
        output, _ = model(data_tensor,None)
        # output = model(data_tensor)
        output = torch.softmax(output, dim=1)
        output = output.detach().numpy().tolist()[0]
        # 将output前三个元素各加 output 最后一个元素的1/3
        output[:3] = [x + output[-1]/3 for x in output[:3]]
        output[:3] = [x/sum(output[:3]) for x in output[:3]]
    return output

def test_game():
    game = TicTacToe()
    game.print_board()
    while game.winner is None:
        player = game.current_player
        print(f"Player {player}, it's your turn.")
        move = input("Enter your move (x, y): ")
        # try:
        x, y = move.split(",")
        x, y = int(x), int(y)
        game.make_move(x,y)
        game.check_win()
        game.print_board()
        if game.winner is not None:
            break
        idx = game.current_player
        # 机器人开始走棋
        actions = game.get_legal_actions()
        score = -2
        act = None
        rate = None
        for action in actions:
             new_game = game.copy()
             new_game.make_move(action[0],action[1])
             rate = analysis(new_game.board)
             p = rate[1]
             a = rate[2]
             sc = a-p
             if sc > score:
                act = action
                score = sc
        if act is not None:
            print(f'平局:{rate[0]:.4f},玩家胜率:{rate[1]:.4f}, ai胜率:{rate[2]:.4f}')
            game.make_move(act[0],act[1])
            game.print_board()
            game.check_win()
        # except:
        #     print("Invalid move. Try again.")
        #     game.print_board()
        #     continue
    game.print_board()
    print(f"Game over. Winner: {game.winner}")

def test_loss():
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(10):
        out_put = tensor = torch.zeros((1, 4))
        out_put[0][1] = 1
        label = torch.tensor([3])
        loss = criterion(out_put, label)
        print(f'out_put:{out_put},label:{label}->{loss}')


if __name__ == "__main__":
    # print(f'{torch.randn(1, 3, 3, 3)}')
    out = analysis([
        [1,2,0],
        [2,1,1],
        [1,2,2]
    ])
    print(f'{out}')
    # test_loss()
    # test_game()
    