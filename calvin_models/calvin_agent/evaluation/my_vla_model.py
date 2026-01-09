"""
统一的 VLA 模型接口，支持多种模型
"""
import numpy as np
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from typing import Literal


class MyVLAModel(CalvinBaseModel):
    def __init__(self,
                 model_type: Literal["random", "openvla", "rt2", "octo", "qwen2vl"] = "openvla",
                 model_path: str = None,
                 device: str = "cuda"):
        """
        初始化VLA模型

        Args:
            model_type: 模型类型 ("random", "openvla", "rt2", "octo", "qwen2vl")
            model_path: 模型路径或HuggingFace模型ID
            device: 运行设备
        """
        self.model_type = model_type
        self.device = device

        print(f"🚀 加载 {model_type.upper()} 模型...")

        if model_type == "random":
            self._init_random()
        elif model_type == "openvla":
            self._init_openvla(model_path or "openvla/openvla-7b")
        elif model_type == "rt2":
            self._init_rt2(model_path or "google/rt-2-base")
        elif model_type == "octo":
            self._init_octo(model_path or "octo-base")
        elif model_type == "qwen2vl":
            self._init_qwen2vl(model_path or "Qwen/Qwen2-VL-8B-Instruct")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        print(f"✅ {model_type.upper()} 模型加载完成")

    def _init_random(self):
        """初始化随机动作模型（基线）"""
        self.model = None
        self.action_bounds = {
            'xyz': (-0.02, 0.02),      # 位置增量
            'rpy': (-0.05, 0.05),      # 旋转增量
            'gripper': [-1, 1]         # 夹爪开合
        }
        print("  使用随机动作作为基线模型")

    def _init_openvla(self, model_path: str):
        """初始化 OpenVLA 模型"""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def _init_rt2(self, model_path: str):
        """初始化 RT-2 模型"""
        from transformers import RT2ForConditionalGeneration, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = RT2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def _init_octo(self, model_path: str):
        """初始化 Octo 模型"""
        try:
            from octo.model.octo_model import OctoModel
            import jax

            self.model = OctoModel.load_pretrained(model_path)
            print(f"  使用设备: {jax.devices()}")
        except ImportError:
            raise ImportError(
                "请安装 Octo: pip install octo-models"
            )

    def _init_qwen2vl(self, model_path: str):
        """初始化 Qwen2-VL 模型（需要微调适配动作输出）"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Qwen2-VL 需要额外的动作解码头（假设已微调）
        # 或使用提示工程从文本生成动作

    def reset(self):
        """重置状态"""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'reset'):
                self.model.reset()

    def step(self, obs, goal):
        """
        推理一步

        Args:
            obs: 环境观察
            goal: 语言指令（字符串）

        Returns:
            action: (7,) numpy array
        """
        rgb_static = obs['rgb_obs']['rgb_static']  # (200, 200, 3)
        rgb_gripper = obs['rgb_obs']['rgb_gripper']  # (84, 84, 3)
        robot_state = obs['robot_obs']  # (15,)

        if self.model_type == "random":
            return self._step_random(rgb_static, rgb_gripper, robot_state, goal)
        elif self.model_type == "openvla":
            return self._step_openvla(rgb_static, rgb_gripper, robot_state, goal)
        elif self.model_type == "rt2":
            return self._step_rt2(rgb_static, rgb_gripper, robot_state, goal)
        elif self.model_type == "octo":
            return self._step_octo(rgb_static, rgb_gripper, robot_state, goal)
        elif self.model_type == "qwen2vl":
            return self._step_qwen2vl(rgb_static, rgb_gripper, robot_state, goal)

    def _step_random(self, rgb_static, rgb_gripper, robot_state, goal):
        """随机动作基线"""
        # 生成随机动作
        action = np.zeros(7)

        # xyz 位置增量 (前3维)
        action[:3] = np.random.uniform(
            self.action_bounds['xyz'][0],
            self.action_bounds['xyz'][1],
            3
        )

        # rpy 旋转增量 (中间3维)
        action[3:6] = np.random.uniform(
            self.action_bounds['rpy'][0],
            self.action_bounds['rpy'][1],
            3
        )

        # 夹爪开合 (最后1维)
        action[6] = np.random.choice(self.action_bounds['gripper'])

        return action

    def _step_openvla(self, rgb_static, rgb_gripper, robot_state, goal):
        """OpenVLA 推理"""
        import torch
        from PIL import Image

        # OpenVLA 使用静态相机图像
        image = Image.fromarray(rgb_static.astype(np.uint8))

        # 准备输入
        prompt = f"In: What action should the robot take to {goal}?\nOut:"
        inputs = self.processor(prompt, image).to(
            self.device,
            dtype=torch.bfloat16
        )

        # 推理
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")

        return action.cpu().numpy()

    def _step_rt2(self, rgb_static, rgb_gripper, robot_state, goal):
        """RT-2 推理"""
        import torch
        from PIL import Image

        # RT-2 通常使用静态相机
        image = Image.fromarray(rgb_static.astype(np.uint8))

        # 准备输入
        inputs = self.processor(
            text=goal,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            # RT-2 输出需要解码为动作
            action = self._decode_rt2_action(outputs)

        return action

    def _step_octo(self, rgb_static, rgb_gripper, robot_state, goal):
        """Octo 推理"""
        import jax.numpy as jnp

        # Octo 使用多视角图像
        observation = {
            "image_primary": rgb_static,
            "image_wrist": rgb_gripper,
            "proprio": robot_state[:7]  # 机器人状态
        }

        # 准备任务
        task = self.model.create_tasks(texts=[goal])

        # 推理
        action = self.model.sample_actions(
            observation,
            task,
            rng=jax.random.PRNGKey(0)
        )

        return np.array(action[0])

    def _step_qwen2vl(self, rgb_static, rgb_gripper, robot_state, goal):
        """Qwen2-VL 推理（需要特殊处理）"""
        import torch
        from PIL import Image

        # Qwen2-VL 主要用于视觉理解，需要微调或提示工程
        image = Image.fromarray(rgb_static.astype(np.uint8))

        # 构建提示（假设模型已微调输出动作）
        messages = [
            {
                "role": "system",
                "content": """You are a precise robot arm controller. You MUST output ONLY 7 numbers separated by commas.

        Output format: x,y,z,roll,pitch,yaw,gripper

        Rules:
        - x,y,z: position delta in meters (range: -0.02 to 0.02)
        - roll,pitch,yaw: rotation delta in radians (range: -0.05 to 0.05)  
        - gripper: MUST be exactly 1 (close) or -1 (open), NO other values allowed

        Example output: 0.01,-0.02,0.005,0.0,0.0,0.0,1"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Task: {goal}\n\nOutput the next action as 7 numbers:"}
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
            response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # 从文本解析动作
        action = self._parse_action_from_text(response)
        return action

    def _decode_rt2_action(self, outputs):
        """解码 RT-2 的动作输出"""
        # RT-2 使用离散化的动作空间，需要反量化
        # 这里是简化示例
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        # 解析动作 token 并转换为连续动作
        action = np.random.uniform(-0.02, 0.02, 7)  # 占位实现
        action[-1] = np.random.choice([-1, 1])
        return action

    def _parse_action_from_text(self, text: str):
        """从文本解析动作"""
        try:
            # 尝试提取数字
            import re
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if len(numbers) >= 7:
                action = np.array([float(n) for n in numbers[:7]])
                return action
        except:
            pass

        # 默认随机动作
        action = np.random.uniform(-0.02, 0.02, 7)
        action[-1] = np.random.choice([-1, 1])
        return action


# 使用示例
if __name__ == "__main__":
    # 方式1: 使用随机基线模型（无需GPU）
    print("=" * 50)
    print("测试随机基线模型")
    print("=" * 50)
    model_random = MyVLAModel(model_type="random")

    # 方式2: 使用 OpenVLA
    # model = MyVLAModel(model_type="openvla")

    # 方式3: 使用 RT-2
    # model = MyVLAModel(model_type="rt2")

    # 方式4: 使用 Octo
    # model = MyVLAModel(model_type="octo")

    # 方式5: 使用 Qwen2-VL
    # model = MyVLAModel(model_type="qwen2vl")

    # 模拟观察
    obs = {
        'rgb_obs': {
            'rgb_static': np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
            'rgb_gripper': np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        },
        'robot_obs': np.random.randn(15)
    }

    goal = "pick up the red block"

    # 测试多个步骤
    print(f"\n任务: {goal}\n")
    for i in range(3):
        action = model_random.step(obs, goal)
        print(f"步骤 {i+1} 动作: {action}")
        print(f"  - 位置增量 (xyz): {action[:3]}")
        print(f"  - 旋转增量 (rpy): {action[3:6]}")
        print(f"  - 夹爪: {'关闭' if action[6] > 0 else '打开'}")