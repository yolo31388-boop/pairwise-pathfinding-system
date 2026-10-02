"""
pairwise-pathfinding-system - PathfindingSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class PathfindingSystem:
    def __init__(self):
        self.state = {}

    def find_path(self, *args, **kwargs):
        """BUG: A*寻路的启发函数用错了，八方向移动用曼哈顿距离"""
        return None

    def check_obstacle(self, *args, **kwargs):
        """BUG: 障碍物检测只检查中心点，不检查整个角色体积"""
        return None

    def build_navigation_mesh(self, *args, **kwargs):
        """BUG: 导航网格不做连通性检查，有的区域走不到"""
        return None

    def request_path(self, *args, **kwargs):
        """BUG: 寻路请求不做异步，主线程等待寻路完成"""
        return None

    def follow_path(self, *args, **kwargs):
        """BUG: 路径跟随不做到达判断，到了路径点还在往前挤"""
        return None

