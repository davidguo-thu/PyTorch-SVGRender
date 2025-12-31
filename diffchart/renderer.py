import torch
import torch.nn as nn
import pydiffvg
from typing import List
from .primitives import ChartElement

class DiffChartRenderer(nn.Module):
    """
    可微图表渲染器。
    管理图表元素列表，并在 forward 中动态构建场景。
    """
    def __init__(self, canvas_width=256, canvas_height=256, use_gpu=True):
        super().__init__()
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # 初始化 pydiffvg 环境
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        pydiffvg.set_use_gpu(torch.cuda.is_available() and use_gpu)
        pydiffvg.set_device(device)
        self.device = device
        
        # 元素列表 (nn.ModuleList 确保参数注册到优化器)
        self.elements = nn.ModuleList()

    def add_element(self, element: ChartElement):
        self.elements.append(element)
        # 确保新加入的元素在正确的设备上
        element.to(self.device)

    def set_elements(self, elements: List[ChartElement]):
        self.elements = nn.ModuleList(elements)
        self.to(self.device)

    def forward(self, background_color=None):
        """
        执行一次完整的渲染流程。
        """
        shapes = []
        shape_groups = []
        
        # 1. 动态获取所有元素的几何形状
        for i, elem in enumerate(self.elements):
            shape, group = elem() # 调用 ChartElement.forward()
            
            # 统一分配 ID，这对 pydiffvg 渲染至关重要
            # group.shape_ids 必须指向 shapes 列表中的索引
            group.shape_ids = torch.tensor([i], device=self.device)
            
            shapes.append(shape)
            shape_groups.append(group)
        
        # 2. 序列化场景
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups
        )
        
        # 3. 光栅化 (Rasterization)
        # 参数: w, h, num_samples_x, num_samples_y, seed, background, *scene_args
        render = pydiffvg.RenderFunction.apply
        img = render(
            self.canvas_width, 
            self.canvas_height, 
            2, 2, 0, # Super-sampling 2x2
            background_color, 
            *scene_args
        )
        
        # 输出是 RGBA，通常需要根据 alpha 进行背景混合
        return img
