# coding:utf-8

import sys
import numpy as np

n = np.nan
maze = np.array([
    [n,  n,  n,  n,  n,  n,  n,  n,  n],
    [n,  0,  0,  0, -1,  0, -1,  0,  n],
    [n,  0, -1,  0, -1,  0, -1,  0,  n],
    [n, -1, -1,  0,  0,  0,  0,  0,  n],
    [n,  0, -1,  0, -1,  0, -1,  0,  n],
    [n,  0,  0,  0, -1,  0, -1,  0,  n],
    [n,  0, -1,  0,  0,  0, -1, 99,  n],
    [n,  0,  0,  0, -1,  0,  0,  0,  n],
    [n,  n,  n,  n,  n,  n,  n,  n,  n]
])

# 定数
ALPHA = 0.2  # LEARNING RATIO
GAMMA = 0.9  # DISCOUNT RATIO
E_GREEDY_RATIO = 0.2


class Maze(object):
    """ 迷路クラス """
    def __init__(self, field=maze):
        self.field = field

    def show_maze(self, state):
        """ 迷路の状態を表示 """
        x_len, y_len = self.field.shape
        for y in y_len:
            line = []
            for x in x_len:
                label = ""
                if np.isnan(field[y, x]):
                    label = "#"
                elif (x, y) == state:
                    label = "@"
                else:
                    label = str(field[y, x]))

                line.append("{:>2}".format(label))

            print " ".join(line)

    def get_actions(self, point):
        """ 移動可能な座標のリストを取得 """
        x, y = point
        if np.isnan(self.field[y, x]):
            # 壁を指定している
            return None

        around_map = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [(_x, _y) for _x, _y in around_map if not np.isnan(self.field[_y, _x])]

    def get_value(self, point):
        x, y = point
        if np.isnan(self.field[y, x]):
            # 壁を指定している
            return None

        return self.field[y, x]


class Q_Learning(object):
    def __init__(maze):
        self.Qvalue = {}
        self.maze = maze
        self.state = (0, 0)

    def learn_one_episode():
        # 現在の位置(環境)を初期化
        self.state = (0, 0)

        while True:
            # 現在の位置（環境）に応じて行動を決定する
            action = self.choose_action(self.state)

            # 環境と行動を使ってQ値を更新
            self.update(self.state, action)

            # actionを実行
            self.move(action)

            # ゴールについたら１エピソード終了
            # (5, 6) : goal
            if self.state == (5, 6):
                break

    def choose_action():
        raise NotImplementedError()

    def choose_action_greedy():
        raise NotImplementedError()

    def update(state, action):
        raise NotImplementedError()

    def move():
        raise NotImplementedError()

    def get_Qvalue(self, state, action):
        """ Q(s,a)を取得する. s:state, a:action """
        try:
            return self.Qvalue[state][action]
        except KeyError:
            return 0.0

    def set_Qvalue(self, state, action, q_value):
        """ Q値に値を代入する. """
        self.Qvalue.setdefault(state, {})
        self.Qvalue[state][action] = q_value

    def try_maze():
        """ 現在のQ値で迷路に挑戦 """
        # 現在の位置(環境)を初期化
        self.state = (0, 0)
        self.show_maze()

        while True:
            # 現在の位置（環境）に応じて行動を決定する
            action = self.choose_action_greedy(self.state)

            # actionを実行
            self.move(action)
            self.show_maze()

            # ゴールについたら終了
            if self.state == (5, 6):
                break

    def show_qvalue():
        raise NotImplementedError()

    def show_maze():
        """ 現在の迷路の状態を表示 """
        self.maze.show_maze()

if __name__ == "__main__":
    maze = Maze()
    maze.show_maze()

    # Q-Learning
    QL = Q_Learning(maze)
    for i in range(1000):
        QL.learn()

    QL.show_qvalue()
    QL.try_maze()
