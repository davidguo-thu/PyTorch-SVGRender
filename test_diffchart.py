import torch
import torch.optim as optim
from diffchart import DiffChartRenderer, BarElement, PieElement, LineElement, PointElement
import pydiffvg

def main():
    print("Initializing DiffChart Renderer...")
    
    # 1. 创建渲染器
    W, H = 256, 256
    renderer = DiffChartRenderer(canvas_width=W, canvas_height=H)
    
    # 2. 创建图表元素 (初始参数)
    # 柱状图 (x, y, w, h)
    bar = BarElement(x=50, y=150, w=30, h=80, color=torch.tensor([0.8, 0.2, 0.2, 1.0]))
    
    # 饼图 (cx, cy, r, start, sweep)
    # 初始是个小扇形
    pie = PieElement(cx=180, cy=180, radius=40, start_angle=0, sweep_angle=1.5, color=torch.tensor([0.2, 0.8, 0.2, 1.0]))
    
    # 折线 (p1, p2)
    line = LineElement(x1=10, y1=10, x2=100, y2=50, width=3.0, color=torch.tensor([0.2, 0.2, 0.8, 1.0]))
    
    # 散点 (cx, cy, r)
    point = PointElement(cx=200, cy=50, r=10, color=torch.tensor([0.9, 0.9, 0.1, 1.0]))
    
    # 添加到渲染器
    renderer.add_element(bar)
    renderer.add_element(pie)
    renderer.add_element(line)
    renderer.add_element(point)
    
    # 3. 设置优化器
    # 我们希望优化所有元素的参数
    optimizer = optim.Adam(renderer.parameters(), lr=1.0)
    
    print("\nInitial Parameters:")
    print(f"Bar Height: {bar.size[1].item():.2f}")
    print(f"Pie Angle: {pie.sweep_angle.item():.2f}")
    print(f"Point Radius: {point.radius.item():.2f}")
    
    # 4. 模拟优化循环
    print("\nStarting Optimization Loop...")
    for i in range(20):
        optimizer.zero_grad()
        
        # Forward pass: 生成图像
        img = renderer()
        
        # Loss Function: 
        # 假设我们的目标是：
        # 1. 让柱子变高 (Maximize h)
        # 2. 让饼图变成半圆 (Target sweep = 3.14)
        # 3. 让圆点变大 (Maximize r)
        
        loss_bar = -bar.size[1] # Maximize height
        loss_pie = (pie.sweep_angle - 3.14159)**2 # Target PI
        loss_point = -point.radius # Maximize radius
        
        total_loss = loss_bar + loss_pie * 100 + loss_point
        
        # Backward pass
        total_loss.backward()
        
        # Update
        optimizer.step()
        
        if i % 5 == 0:
            print(f"Step {i}: Loss {total_loss.item():.4f}")
            print(f"  -> Bar H: {bar.size[1].item():.2f} (Grad: {bar.size.grad[1].item():.2f})")
            print(f"  -> Pie Sweep: {pie.sweep_angle.item():.2f}")
            
    print("\nFinal Parameters:")
    print(f"Bar Height: {bar.size[1].item():.2f}")
    print(f"Pie Angle: {pie.sweep_angle.item():.2f}")
    print(f"Point Radius: {point.radius.item():.2f}")
    
    # 保存最终结果看看
    pydiffvg.imwrite(img.cpu(), "test_chart_optimized.png", gamma=1.0)
    print("\nSaved result to test_chart_optimized.png")

if __name__ == "__main__":
    main()
