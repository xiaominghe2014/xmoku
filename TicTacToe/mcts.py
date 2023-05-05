import random
import math

# 定义蒙特卡洛树节点
class MonteCarloTreeNode:
    def __init__(self, state, action=None, parent=None, prior=0.):
        self.state = state # 游戏状态
        self.prior = prior # 先验概率
        self.player = state.current_player
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0 # 访问次数
        self.win = 0 # 胜
        self.lose = 0 # 负
        self.draw = 0 # 和
        self.untried_actions = state.get_legal_actions() # 还未尝试过的动作集合
    
    def get_legal_actions(self):
        return self.state.get_legal_actions()

    def add_child(self, action, prior_call):
        st = self.state.copy()
        if self.action is not None:
            st.make_move(action[0],action[1])
            if prior_call is not None:
                self.prior = prior_call(st)[st.last_move_player()]
        node = MonteCarloTreeNode(st, action, self)
        self.untried_actions.remove(action)
        self.children.append(node)
        return node

    def update(self):
        self.visits += 1
        if self.state.winner is not None:
            if self.player==self.state.winner:
                self.lose +=1
            elif self.state.other(self.player)==self.state.winner:
                self.win +=1
            else:
                self.draw +=1

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb())
    
    def ucb(self):
        if self.visits == 0:
            return float('inf')
        return (self.win / self.visits + self.prior) + 1.41 * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_root(self):
        return self.parent is None
    
    def is_leaf(self):
        return len(self.children) == 0


class MonteCarloTreeSearch:
    def __init__(self, state, prior_call):
        self.state = state
        self.prior_call = prior_call
    
    def get_action(self, searchCnt):
        # 构建根节点
        root = MonteCarloTreeNode(self.state)

        for _ in range(searchCnt):
            # 迭代搜索
            node = root
            # 选择
            while node.untried_actions == [] and node.children != []:
                node = node.select_child()
            # 扩展
            if node.untried_actions != []:
                a = random.choice(node.untried_actions)
                node = node.add_child(a, self.prior_call)
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
            