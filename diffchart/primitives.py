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
        self.color = nn.Parameter(color)

    def forward(self):
        """
        返回用于 pydiffvg 渲染的 (shape, shape_group) 列表
        注意：为了支持复合图形（如 Boxplot），返回值统一为 List[shape] 和 List[group]
        或者单个 shape/group，Renderer 会自动处理。
        """
        raise NotImplementedError

# ==================== 基础元素 ====================

class BarElement(ChartElement):
    def __init__(self, x, y, w, h, color=None):
        super().__init__(color)
        self.pos = nn.Parameter(torch.tensor([float(x), float(y)]))
        self.size = nn.Parameter(torch.tensor([float(w), float(h)]))

    def forward(self):
        shape = pydiffvg.Rect(p_min=self.pos, p_max=self.pos + self.size)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group

class PointElement(ChartElement):
    def __init__(self, cx, cy, r, color=None):
        super().__init__(color)
        self.center = nn.Parameter(torch.tensor([float(cx), float(cy)]))
        self.radius = nn.Parameter(torch.tensor(float(r)))

    def forward(self):
        shape = pydiffvg.Circle(radius=self.radius, center=self.center)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group

class LineElement(ChartElement):
    def __init__(self, x1, y1, x2, y2, width=2.0, color=None):
        super().__init__(color)
        self.p1 = nn.Parameter(torch.tensor([float(x1), float(y1)]))
        self.p2 = nn.Parameter(torch.tensor([float(x2), float(y2)]))
        self.width = nn.Parameter(torch.tensor(float(width)))

    def forward(self):
        points = torch.stack([self.p1, self.p2], dim=0)
        num_control_points = torch.tensor([0])
        shape = pydiffvg.Path(
            num_control_points=num_control_points, points=points,
            is_closed=False, stroke_width=self.width
        )
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=self.color)
        return shape, group

# ==================== 进阶元素 ====================

class PieElement(ChartElement):
    """
    实心扇形
    """
    def __init__(self, cx, cy, radius, start_angle, sweep_angle, color=None, num_segments=30):
        super().__init__(color)
        self.center = nn.Parameter(torch.tensor([float(cx), float(cy)]))
        self.radius = nn.Parameter(torch.tensor(float(radius)))
        self.start_angle = nn.Parameter(torch.tensor(float(start_angle))) 
        self.sweep_angle = nn.Parameter(torch.tensor(float(sweep_angle)))
        self.num_segments = num_segments

    def forward(self):
        device = self.center.device
        t = torch.linspace(0, 1, self.num_segments, device=device)
        thetas = self.start_angle + t * self.sweep_angle
        
        arc_x = self.center[0] + self.radius * torch.cos(thetas)
        arc_y = self.center[1] + self.radius * torch.sin(thetas)
        
        arc_points = torch.stack([arc_x, arc_y], dim=1)
        center_point = self.center.unsqueeze(0)
        points = torch.cat([center_point, arc_points, center_point], dim=0)
        
        shape = pydiffvg.Polygon(points=points, is_closed=True)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group

class DonutElement(ChartElement):
    """
    环形图 (甜甜圈)
    实现逻辑：构建一个闭合多边形，路径为 外圈(顺时针) -> 内圈(逆时针)
    参数: cx, cy, inner_radius, outer_radius, start_angle, sweep_angle
    """
    def __init__(self, cx, cy, r_in, r_out, start_angle, sweep_angle, color=None, num_segments=30):
        super().__init__(color)
        self.center = nn.Parameter(torch.tensor([float(cx), float(cy)]))
        self.r_in = nn.Parameter(torch.tensor(float(r_in)))
        self.r_out = nn.Parameter(torch.tensor(float(r_out)))
        self.start_angle = nn.Parameter(torch.tensor(float(start_angle)))
        self.sweep_angle = nn.Parameter(torch.tensor(float(sweep_angle)))
        self.num_segments = num_segments

    def forward(self):
        device = self.center.device
        t = torch.linspace(0, 1, self.num_segments, device=device)
        thetas = self.start_angle + t * self.sweep_angle
        
        # 外圈 (Outer Circle)
        out_x = self.center[0] + self.r_out * torch.cos(thetas)
        out_y = self.center[1] + self.r_out * torch.sin(thetas)
        outer_points = torch.stack([out_x, out_y], dim=1)
        
        # 内圈 (Inner Circle) - 需要反向 (flip) 以形成正确的孔洞拓扑
        in_x = self.center[0] + self.r_in * torch.cos(thetas)
        in_y = self.center[1] + self.r_in * torch.sin(thetas)
        inner_points = torch.stack([in_x, in_y], dim=1)
        inner_points = torch.flip(inner_points, dims=[0]) # 反转顺序
        
        # 缝合: 外圈 -> 内圈 -> 回到起点
        points = torch.cat([outer_points, inner_points, outer_points[:1]], dim=0)
        
        shape = pydiffvg.Polygon(points=points, is_closed=True)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group

class AreaElement(ChartElement):
    """
    面积图 (Filled Area)
    参数: xs (X轴坐标序列), ys (Y轴坐标序列), base_y (基准线Y坐标)
    注意：这里 xs 和 ys 通常是定长的 tensor，如果需要变长比较麻烦，通常设定最大长度然后mask。
    为了简化，这里假设输入是固定数量的控制点。
    """
    def __init__(self, xs, ys, base_y, color=None):
        super().__init__(color)
        # 假设 xs, ys 是 List 或者 Tensor
        self.xs = nn.Parameter(torch.as_tensor(xs, dtype=torch.float32))
        self.ys = nn.Parameter(torch.as_tensor(ys, dtype=torch.float32))
        self.base_y = nn.Parameter(torch.tensor(float(base_y)))

    def forward(self):
        # 构建多边形: (x0, y0) -> (x1, y1) ... -> (xn, yn) -> (xn, base) -> (x0, base)
        
        top_points = torch.stack([self.xs, self.ys], dim=1)
        
        # 底部右下角点
        bottom_right = torch.stack([self.xs[-1], self.base_y], dim=0).unsqueeze(0)
        # 底部左下角点
        bottom_left = torch.stack([self.xs[0], self.base_y], dim=0).unsqueeze(0)
        
        points = torch.cat([top_points, bottom_right, bottom_left], dim=0)
        
        shape = pydiffvg.Polygon(points=points, is_closed=True)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group

class BoxplotElement(ChartElement):
    """
    箱线图 (Box-and-Whisker Plot)
    这是一个复合元素，由一个矩形和两条线组成。
    参数: x (中心), box_y1 (箱底), box_y2 (箱顶), whisker_y1 (下须), whisker_y2 (上须), width (箱宽)
    """
    def __init__(self, x, box_y1, box_y2, whisker_y1, whisker_y2, width, color=None, stroke_color=None):
        super().__init__(color) # 这是箱体填充色
        
        if stroke_color is None:
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]) # 默认黑色线条
        self.stroke_color = nn.Parameter(stroke_color)
        
        self.x = nn.Parameter(torch.tensor(float(x)))
        self.width = nn.Parameter(torch.tensor(float(width)))
        
        # Y轴坐标
        self.box_y1 = nn.Parameter(torch.tensor(float(box_y1)))
        self.box_y2 = nn.Parameter(torch.tensor(float(box_y2)))
        self.whisker_y1 = nn.Parameter(torch.tensor(float(whisker_y1)))
        self.whisker_y2 = nn.Parameter(torch.tensor(float(whisker_y2)))
        
        self.line_width = nn.Parameter(torch.tensor(2.0))

    def forward(self):
        shapes = []
        groups = []
        
        half_w = self.width / 2
        
        # 1. 箱体 (Rectangle)
        # p_min = (x - w/2, box_y1)
        # p_max = (x + w/2, box_y2)
        # 注意：这里假设 y1 < y2，如果是坐标系向下，可能需要 min/max 处理
        p_min = torch.stack([self.x - half_w, torch.min(self.box_y1, self.box_y2)])
        p_max = torch.stack([self.x + half_w, torch.max(self.box_y1, self.box_y2)])
        
        box_shape = pydiffvg.Rect(p_min=p_min, p_max=p_max)
        box_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color, stroke_color=self.stroke_color)
        
        shapes.append(box_shape)
        groups.append(box_group)
        
        # 2. 中线 (从 whisker_y1 到 box_y_min)
        # 3. 中线 (从 box_y_max 到 whisker_y2)
        # 4. 须帽 (Cap at whisker_y1)
        # 5. 须帽 (Cap at whisker_y2)
        
        # 为了简化，我们用 Path 画这所有的线
        # 下须线
        y_min = torch.min(self.box_y1, self.box_y2)
        y_max = torch.max(self.box_y1, self.box_y2)
        
        # 线段: (x, whisker_y1) -> (x, y_min)
        line_bot_points = torch.stack([
            torch.stack([self.x, self.whisker_y1]),
            torch.stack([self.x, y_min])
        ])
        shapes.append(pydiffvg.Path(
            num_control_points=torch.tensor([0]), points=line_bot_points, is_closed=False, stroke_width=self.line_width))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=self.stroke_color))

        # 线段: (x, y_max) -> (x, whisker_y2)
        line_top_points = torch.stack([
            torch.stack([self.x, y_max]),
            torch.stack([self.x, self.whisker_y2])
        ])
        shapes.append(pydiffvg.Path(
            num_control_points=torch.tensor([0]), points=line_top_points, is_closed=False, stroke_width=self.line_width))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=self.stroke_color))

        # 须帽 (横线)
        cap_w = half_w * 0.5 # 须帽通常比箱子窄一点
        
        # Bottom Cap
        cap_bot_points = torch.stack([
            torch.stack([self.x - cap_w, self.whisker_y1]),
            torch.stack([self.x + cap_w, self.whisker_y1])
        ])
        shapes.append(pydiffvg.Path(
            num_control_points=torch.tensor([0]), points=cap_bot_points, is_closed=False, stroke_width=self.line_width))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=self.stroke_color))
        
        # Top Cap
        cap_top_points = torch.stack([
            torch.stack([self.x - cap_w, self.whisker_y2]),
            torch.stack([self.x + cap_w, self.whisker_y2])
        ])
        shapes.append(pydiffvg.Path(
            num_control_points=torch.tensor([0]), points=cap_top_points, is_closed=False, stroke_width=self.line_width))
        groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=self.stroke_color))

        return shapes, groups

class StripElement(ChartElement):
    """
    条带图 (Strip Plot) / 误差带 (Error Band)
    两条曲线中间填充区域
    参数: xs, y_upper, y_lower
    """
    def __init__(self, xs, y_upper, y_lower, color=None):
        super().__init__(color)
        self.xs = nn.Parameter(torch.as_tensor(xs, dtype=torch.float32))
        self.y_upper = nn.Parameter(torch.as_tensor(y_upper, dtype=torch.float32))
        self.y_lower = nn.Parameter(torch.as_tensor(y_lower, dtype=torch.float32))

    def forward(self):
        # 构造路径：Upper 曲线 (正向) -> Lower 曲线 (反向) -> 闭合
        
        upper_points = torch.stack([self.xs, self.y_upper], dim=1)
        
        lower_points = torch.stack([self.xs, self.y_lower], dim=1)
        lower_points = torch.flip(lower_points, dims=[0])
        
        points = torch.cat([upper_points, lower_points], dim=0)
        
        shape = pydiffvg.Polygon(points=points, is_closed=True)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=self.color)
        return shape, group
