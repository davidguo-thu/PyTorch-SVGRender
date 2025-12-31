import torch
import torch.nn as nn
import pydiffvg
import math

class ChartElement(nn.Module):
    """
    所有图表元素的基类。
    每个元素都包含语义参数（位置、大小、数值等）和外观参数（颜色）。
    """
    def __init__(self, color=None):
        super().__init__()
        # 默认灰色
        if color is None:
            color = torch.tensor([0.5, 0.5, 0.5, 1.0])
        
        # 颜色是可优化的参数
        self.color = nn.Parameter(color)

    def forward(self):
        """
        返回用于 pydiffvg 渲染的 (shape, shape_group)
        """
        raise NotImplementedError

class BarElement(ChartElement):
    """
    柱状图元素 (Rectangle)
    语义参数: x, y (左上角), w, h (宽, 高)
    """
    def __init__(self, x, y, w, h, color=None):
        super().__init__(color)
        self.pos = nn.Parameter(torch.tensor([float(x), float(y)]))
        self.size = nn.Parameter(torch.tensor([float(w), float(h)]))

    def forward(self):
        # 确保宽高为正数 (虽然优化过程中可能会出现负数，但在渲染前最好处理一下)
        # 这里直接使用参数，让损失函数去约束非负性是更常用的做法
        
        p_min = self.pos
        p_max = self.pos + self.size
        
        # pydiffvg.Rect 接受 p_min 和 p_max
        shape = pydiffvg.Rect(p_min=p_min, p_max=p_max)
        
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]), # ID 稍后在 Renderer 中统一分配
            fill_color=self.color
        )
        return shape, group

class PointElement(ChartElement):
    """
    散点图元素 (Circle)
    语义参数: cx, cy (圆心), r (半径/大小)
    """
    def __init__(self, cx, cy, r, color=None):
        super().__init__(color)
        self.center = nn.Parameter(torch.tensor([float(cx), float(cy)]))
        self.radius = nn.Parameter(torch.tensor(float(r)))

    def forward(self):
        shape = pydiffvg.Circle(
            radius=self.radius,
            center=self.center
        )
        
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=self.color
        )
        return shape, group

class LineElement(ChartElement):
    """
    折线图/连线元素 (Polyline/Path)
    语义参数: p1(x,y), p2(x,y), width (线宽)
    """
    def __init__(self, x1, y1, x2, y2, width=2.0, color=None):
        super().__init__(color)
        self.p1 = nn.Parameter(torch.tensor([float(x1), float(y1)]))
        self.p2 = nn.Parameter(torch.tensor([float(x2), float(y2)]))
        self.width = nn.Parameter(torch.tensor(float(width)))

    def forward(self):
        # 使用 Path 来绘制线段
        points = torch.stack([self.p1, self.p2], dim=0)
        
        # Path 需要 num_control_points，对于直线，每个段有 0 个控制点
        num_control_points = torch.tensor([0])
        
        shape = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
            stroke_width=self.width
        )
        
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=None, # 线条通常没有填充
            stroke_color=self.color
        )
        return shape, group

class PieElement(ChartElement):
    """
    饼图/环形图元素 (Sector/Wedge)
    语义参数: center(x,y), radius, start_angle, sweep_angle
    """
    def __init__(self, cx, cy, radius, start_angle, sweep_angle, color=None, num_segments=30):
        super().__init__(color)
        self.center = nn.Parameter(torch.tensor([float(cx), float(cy)]))
        self.radius = nn.Parameter(torch.tensor(float(radius)))
        self.start_angle = nn.Parameter(torch.tensor(float(start_angle))) # 弧度
        self.sweep_angle = nn.Parameter(torch.tensor(float(sweep_angle))) # 弧度
        self.num_segments = num_segments # 近似精度，不可导常量

    def forward(self):
        # 核心：使用多边形逼近扇形，确保所有计算链都是 PyTorch Operation
        
        # 1. 生成 [0, 1] 的插值因子
        device = self.center.device
        t = torch.linspace(0, 1, self.num_segments, device=device)
        
        # 2. 计算圆弧上的角度序列
        # theta = start + t * sweep
        thetas = self.start_angle + t * self.sweep_angle
        
        # 3. 极坐标转笛卡尔坐标
        # x = cx + r * cos(theta)
        # y = cy + r * sin(theta)
        arc_x = self.center[0] + self.radius * torch.cos(thetas)
        arc_y = self.center[1] + self.radius * torch.sin(thetas)
        
        arc_points = torch.stack([arc_x, arc_y], dim=1)
        
        # 4. 构建闭合多边形: 圆心 -> 弧线点 -> 圆心
        center_point = self.center.unsqueeze(0)
        points = torch.cat([center_point, arc_points, center_point], dim=0)
        
        shape = pydiffvg.Polygon(points=points, is_closed=True)
        
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=self.color
        )
        
        return shape, group
