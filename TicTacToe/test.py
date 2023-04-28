
from ticTacToe import TicTacToe

import sys

from net import CNNModel
import torch
import numpy as np
# 加载模型
# 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel()
model.load_state_dict(torch.load('model/model.pth'))

# 将模型设置为评估模式
model.eval()

def analysis(board):
    data_tensor = torch.Tensor(board).unsqueeze(0).unsqueeze(0)
    # print(f'data_tensor: {data_tensor}')
    output = model(data_tensor)
    output = torch.softmax(output, dim=1)
    output = output.detach().numpy().tolist()[0]
    # print(f' 预测结果 {output}')
    return output

if __name__ == "__main__":
    # analysis([
    #     [0,1,0],
    #     [0,1,0],
    #     [0,1,0]
    # ])
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
             print(f'rate:{rate}')
             p = rate[1]
             a = rate[2]
             sc = a-p
             print(f'sc:{sc}')
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