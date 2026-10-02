import pytest
from pathfinding.system import PathfindingSystem


class TestPathfindingSystem:

    def test_find_path(self):
        """A*寻路的启发函数用错了，八方向移动用曼哈顿距离"""
        obj = PathfindingSystem()
        result = obj.find_path()
        self.assertIsNotNone(result)

    def test_check_obstacle(self):
        """障碍物检测只检查中心点，不检查整个角色体积"""
        obj = PathfindingSystem()
        result = obj.check_obstacle()
        self.assertIsNotNone(result)

    def test_build_navigation_mesh(self):
        """导航网格不做连通性检查，有的区域走不到"""
        obj = PathfindingSystem()
        result = obj.build_navigation_mesh()
        self.assertIsNotNone(result)

    def test_request_path(self):
        """寻路请求不做异步，主线程等待寻路完成"""
        obj = PathfindingSystem()
        result = obj.request_path()
        self.assertIsNotNone(result)

    def test_follow_path(self):
        """路径跟随不做到达判断，到了路径点还在往前挤"""
        obj = PathfindingSystem()
        result = obj.follow_path()
        self.assertIsNotNone(result)

