import copy
from mcts import MonteCarloTreeSearch

class TicTacToe:
    def __init__(self, prior_call=None):
        self.current_player = 1
        self.winner = None
        self.move_history = [] # 添加一个实例变量，用于保存走子步骤
        self.w = 3
        self.h = 3
        self.board = [[0 for j in range(self.h)] for i in range(self.w)]
        self.mcts = MonteCarloTreeSearch(self, prior_call)

    def opponet(self):
        if self.current_player == 1:
            return 2
        return 1

    def other(self, p):
        if p == 1:
            return 2
        return 1

    def copy(self):
        game = TicTacToe()
        game.current_player = self.current_player
        game.winner = self.winner
        game.move_history = copy.deepcopy(self.move_history)
        game.board = copy.deepcopy(self.board)
        return game

    def print_board(self):
        print('-------------')
        for y in range(self.h):
            print('|', self.cell_char_at(0,y), '|', self.cell_char_at(1,y), '|', self.cell_char_at(2,y), '|')
            print('-------------')

    def copy_board(self):
        return copy.deepcopy(self.board)

    def cell_char_at(self, x, y):
        v = self.board[x][y]
        if v==0:
            return ' '
        if v==1:
            return 'X'
        if v==2:
            return 'O'
        else:
            print(f'Invalid {x},{y}')

    def can_move(self, x ,y):
        return self.board[x][y] == 0

    def make_move(self, x, y):
        if self.board[x][y] == 0:
            self.board[x][y] = self.current_player
            self.move_history.append((x, y))
            self.current_player = self.opponet()
            return True
        else:
            print(f'Invalid move ({x},{y})')
            self.print_board()
            return False

    def undo_move(self):
        if len(self.move_history) > 0:
            x, y = self.move_history.pop()
            self.board[x][y] = 0
            if self.current_player == 1:
                self.current_player = 2
            else:
                self.current_player = 1
            return True
        else:
            print('No move to undo')
            return False
    
    def last_move_player(self):
        return self.opponet()

    def check_win(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.board[i][0]
                return self.winner
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                self.winner = self.board[0][i]
                return self.winner
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.winner = self.board[0][0]
            return self.winner
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.board[0][2]
            return self.winner
        if all(0 not in row for row in self.board):
            print(f"平局")
            self.winner = 0
            return self.winner
        return None
    
    def get_legal_actions(self):
        actions = []
        for x in range(self.w):
            for y in range(self.h):
                if self.board[x][y] == 0:
                    actions.append((x, y))
        return actions
    
    def get_best_action(self):
        action = self.mcts.get_action(100)
        return action

    def get_train_data(self):
        if self.winner is None:
            return (self.board, 3)
        return (self.board, self.winner)


    def rate_check(self, rate):
        if rate[0]> 0.99:
            self.winner = 0
        elif rate[1]> 0.99:
            self.winner = 1
        elif rate[2]> 0.99:
            self.winner = 2
