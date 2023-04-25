from gomoku import Gomoku

game = Gomoku()
game.print_board()
while True:
    
    # 获取当前玩家的下一步动作
    action = game.get_best_action()
    print(action)
    # 将下一步动作添加到棋盘上
    game.play(action[0], action[1])
    game.print_board()
    # 判断是否有玩家获胜
    if game.check_win(action[0], action[1]):
        break