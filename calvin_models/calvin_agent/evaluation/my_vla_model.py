"""
我的 VLA 模型接口
"""
import numpy as np
from calvin_agent.models.calvin_base_model import CalvinBaseModel


class MyVLAModel(CalvinBaseModel):
    def __init__(self):
        print("🚀 加载我的VLA模型...")
        # TODO: 加载你的模型
        # self.model = ...
        pass

    def reset(self):
        """重置状态"""
        pass

    def step(self, obs, goal):
        """
        推理一步

        Args:
            obs: 环境观察
            goal: 语言指令（字符串）

        Returns:
            action: (7,) numpy array
        """
        # 🔥 你的推理代码
        rgb_static = obs['rgb_obs']['rgb_static']  # (200, 200, 3)
        rgb_gripper = obs['rgb_obs']['rgb_gripper']  # (84, 84, 3)
        robot_state = obs['robot_obs']  # (15,)

        # action = self.model.predict(rgb_static, goal)

        # 临时随机动作
        action = np.random.uniform(-0.02, 0.02, 7)
        action[-1] = np.random.choice([-1, 1])

        return action