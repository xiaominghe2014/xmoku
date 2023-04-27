
from ticTacToe import TicTacToe

import sys

from net import LSTMModel
import torch
import numpy as np
# 加载模型
# 实例化模型
input_size = 3
hidden_size = 128
num_layers = 1
num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('model/model.pth'))

# 将模型设置为评估模式
model.eval()

def analysis(board):
    data_tensor = torch.Tensor(board)
    print(f'data_tensor: {data_tensor}')
    output = model(data_tensor)
    output = torch.softmax(output, dim=1)
    print(f' 预测结果 {output}')

if __name__ == "__main__":
    # analysis([[
    #     [0,1,0],
    #     [0,1,0],
    #     [0,1,0]
    # ]])
    game = TicTacToe()
    game.print_board()
    while game.winner is None:
        player = game.current_player
        print(f"Player {player}, it's your turn.")
        move = input("Enter your move (x, y): ")
        try:
            x, y = move.split(",")
            x, y = int(x), int(y)
            game.make_move(x,y)
            analysis([game.board])
            game.check_win()
            game.print_board()
        except:
            print("Invalid move. Try again.")
            game.print_board()
            continue
    game.print_board()
    print(f"Game over. Winner: {game.winner}")