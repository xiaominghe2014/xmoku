import random
import math
import copy

class MonteCarloTreeSearch:
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.root = Node(board, player)

    def get_action(self):
        # 构建根节点
        root = Node(self.board, self.player)

        # 迭代搜索
        for i in range(1000):
            node = root
            board = copy.deepcopy(self.board)

            # 选择
            while node.untried_actions == [] and node.children != []:
                node = node.select_child()
                board[node.action[0]][node.action[1]] = node.player

            # 扩展
            if node.untried_actions != []:
                a = random.choice(node.untried_actions)
                board[a[0]][a[1]] = node.player
                node = node.add_child(a, board, 3 - node.player)

            # 模拟
            while True:
                actions = [(i, j) for i in range(15) for j in range(15) if board[i][j] == 0]
                if actions == []:
                    break
                a = random.choice(actions)
                board[a[0]][a[1]] = node.player
                node.player = 3 - node.player

            # 回溯
            while node != None:
                node.update()
                node = node.parent

        # 返回最佳动作
        if not root.children:
        # 返回默认值
            return None
        else:
            return max(root.children, key=lambda c: c.visits).action

    def update_with_move(self, move):
        x, y = move
        for child in self.root.children:
            if child.action == (x, y):
                self.root = child
                self.root.parent = None
                return
        new_board = copy.deepcopy(self.board)
        new_board[x][y] = self.player
        self.root = Node(new_board, 3 - self.player)

class Node:
    def __init__(self, board, player, action=None, parent=None):
        self.board = board
        self.player = player
        self.action = action
        self.parent = parent
        self.children = []
        self.untried_actions = [(i, j) for i in range(15) for j in range(15) if board[i][j] == 0]
        self.wins = 0
        self.visits = 0

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb())

    def add_child(self, action, board, player):
        node = Node(board, player, action, self)
        self.untried_actions.remove(action)
        self.children.append(node)
        return node

    def update(self):
        self.visits += 1
        self.wins += int(self.is_winner(self.player))

    def is_winner(self, player):
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == player:
                    if i + 4 < 15 and self.board[i+1][j] == player and self.board[i+2][j] == player and self.board[i+3][j] == player and self.board[i+4][j] == player:
                        return True
                    if j + 4 < 15 and self.board[i][j+1] == player and self.board[i][j+2] == player and self.board[i][j+3] == player and self.board[i][j+4] == player:
                        return True
                    if i + 4 < 15 and j + 4 < 15 and self.board[i+1][j+1] == player and self.board[i+2][j+2] == player and self.board[i+3][j+3] == player and self.board[i+4][j+4] == player:
                        return True
                    if i + 4 < 15 and j - 4 >= 0 and self.board[i+1][j-1] == player and self.board[i+2][j-2] == player and self.board[i+3][j-3] == player and self.board[i+4][j-4] == player:
                        return True
        return False

    def ucb(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + 1.41 * math.sqrt(math.log(self.parent.visits) / self.visits)