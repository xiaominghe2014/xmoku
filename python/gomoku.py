
from mcts import MonteCarloTreeSearch

class Gomoku:
    def __init__(self):
        # 初始化棋盘
        self.board = [[0 for _ in range(15)] for _ in range(15)]
        # 初始化当前玩家
        self.current_player = 1
        # 初始化蒙特卡洛树搜索
        self.mcts = MonteCarloTreeSearch(self.board, self.current_player)

    def play(self, x, y):
        print(f"Player {self.current_player} !")
        print(f"{self.board[x][y]}")
        # 判断该位置是否为空
        if self.board[x][y] != 0:
            return False
        # 下棋
        self.board[x][y] = self.current_player
        
        # 判断是否胜利
        if self.check_win(x, y):
            print(f"Player {self.current_player} wins!")
            return True
        # 切换玩家
        self.current_player = 3 - self.current_player
        # 更新蒙特卡洛树搜索
        self.mcts.update_with_move((x, y))
        return True

    def check_win(self, x, y):
        # 判断横向是否有五子相连
        if self.count(x, y, 1, 0) + self.count(x, y, -1, 0) >= 6:
            return True
        # 判断纵向是否有五子相连
        if self.count(x, y, 0, 1) + self.count(x, y, 0, -1) >= 6:
            return True
        # 判断左上到右下是否有五子相连
        if self.count(x, y, 1, 1) + self.count(x, y, -1, -1) >= 6:
            return True
        # 判断右上到左下是否有五子相连
        if self.count(x, y, 1, -1) + self.count(x, y, -1, 1) >= 6:
            return True
        return False

    def count(self, x, y, dx, dy):
        # 从(x,y)出发，沿着(dx,dy)方向走，能够走多远
        if not (0 <= x < 15 and 0 <= y < 15):
            return 0
        if self.board[x][y] != self.current_player:
            return 0
        return 1 + self.count(x+dx, y+dy, dx, dy)
    
    def get_best_action(self):
        # # 获取当前棋盘状态和当前玩家
        # state = self.get_current_state()
        # # 获取当前玩家的下一步动作
        # _, action = self.get_action(state)
        # # 返回下一步动作
        action = self.mcts.get_action()
        return action

    def get_current_state(self):
        return (self.board, self.current_player)
    
    def print_board(self):
        # 打印横坐标
        print("   ", end="")
        for i in range(15):
            print(chr(ord('a') + i), end=" ")
        print()
        # 打印棋盘
        for i in range(15):
            # 打印纵坐标
            print(f"{i+1:2d}", end=" ")
            for j in range(15):
                if self.board[i][j] == 1:
                    print("X", end=" ")
                elif self.board[i][j] == 2:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()
    
    

