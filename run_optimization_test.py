import torch
import torch.optim as optim
import pydiffvg
from pathlib import Path
import math
import random

# 导入我们刚刚实现的库
from diffchart import DiffChartRenderer, BarElement, LineElement, DonutElement

# 结果保存路径
OUTPUT_DIR = Path("optimization_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_image(img_tensor, filename):
    """辅助函数：保存 Tensor 为图片"""
    # img_tensor: [H, W, 4]
    pydiffvg.imwrite(img_tensor.cpu(), str(OUTPUT_DIR / filename), gamma=1.0)

def print_params(name, val_gt, val_opt):
    """辅助函数：打印参数对比"""
    diff = abs(val_gt - val_opt)
    print(f"  {name:10s} | GT: {val_gt:8.2f} | Opt: {val_opt:8.2f} | Diff: {diff:8.2f}")

def run_bar_chart_test():
    print("\n" + "="*50)
    print("TEST CASE 1: Bar Chart Optimization")
    print("="*50)
    
    W, H = 256, 256
    
    # ================= 1. 生成目标图 (Ground Truth) =================
    target_renderer = DiffChartRenderer(W, H)
    
    # 添加坐标轴 (X轴, Y轴)
    axis_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    target_renderer.add_element(LineElement(20, 230, 240, 230, width=2.0, color=axis_color)) # X-axis
    target_renderer.add_element(LineElement(20, 230, 20, 20, width=2.0, color=axis_color))   # Y-axis
    
    # 添加 3 个柱子 (Ground Truth 参数)
    # 假设我们知道有3个柱子，颜色也是已知的(或者作为先验)，但不知道确切的几何参数
    gt_bars = [
        {'x': 40,  'y': 130, 'w': 30, 'h': 100, 'color': torch.tensor([0.8, 0.2, 0.2, 1.0])}, # Red
        {'x': 90,  'y': 80,  'w': 30, 'h': 150, 'color': torch.tensor([0.2, 0.8, 0.2, 1.0])}, # Green
        {'x': 140, 'y': 160, 'w': 30, 'h': 70,  'color': torch.tensor([0.2, 0.2, 0.8, 1.0])}, # Blue
    ]
    
    for b in gt_bars:
        target_renderer.add_element(BarElement(b['x'], b['y'], b['w'], b['h'], color=b['color']))
        
    # 渲染目标图
    print("Rendering Target Image...")
    target_image = target_renderer()
    save_image(target_image, "bar_target.png")
    
    # ================= 2. 初始化待优化模型 (Random Init) =================
    opt_renderer = DiffChartRenderer(W, H)
    
    # 添加坐标轴 (假设坐标轴我们也需要微调，或者固定)
    # 这里我们固定坐标轴，只优化柱子
    opt_renderer.add_element(LineElement(20, 230, 240, 230, width=2.0, color=axis_color))
    opt_renderer.add_element(LineElement(20, 230, 20, 20, width=2.0, color=axis_color))
    
    opt_bars = []
    # 添加 3 个初始状态很糟糕的柱子
    for i, b in enumerate(gt_bars):
        # 随机初始化高度和宽度，位置稍有偏差
        rand_h = random.uniform(20, 50) # 初始都很矮
        rand_w = random.uniform(10, 20) # 初始都很瘦
        rand_x = b['x'] + random.uniform(-10, 10) # 位置稍微偏一点
        rand_y = 230 - rand_h # 这里的y通常是由底边决定的，为了简化，我们让优化器自己去寻找正确的y
        
        # 注意：这里我们让颜色固定为正确颜色，专注于几何参数的优化
        # 如果颜色也需要优化，只需在 BarElement 初始化时不指定 color，或者给个随机 color
        bar = BarElement(rand_x, 150, rand_w, rand_h, color=b['color'])
        opt_renderer.add_element(bar)
        opt_bars.append(bar)

    # 渲染初始图
    print("Rendering Initial Guess...")
    save_image(opt_renderer(), "bar_init.png")

    # ================= 3. 优化循环 =================
    # 我们只优化柱子的参数 (pos, size)，不优化坐标轴
    params_to_optimize = []
    for bar in opt_bars:
        params_to_optimize.extend(bar.parameters())
        
    optimizer = optim.Adam(params_to_optimize, lr=1.0) # 学习率设大一点，像素级优化通常需要较大LR
    
    print("Starting Optimization...")
    for step in range(201):
        optimizer.zero_grad()
        
        # Forward
        rendered_image = opt_renderer()
        
        # Loss: MSE between rendered image and target image
        loss = torch.mean((rendered_image - target_image) ** 2)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")
            save_image(rendered_image, f"bar_step_{step:03d}.png")
            
    print("Optimization Finished.")
    
    # ================= 4. 结果验证 =================
    print("\nParameter Comparison (Height):")
    for i, (gt, opt) in enumerate(zip(gt_bars, opt_bars)):
        print(f"Bar {i}:")
        print_params("Height", gt['h'], opt.size[1].item())
        print_params("Width", gt['w'], opt.size[0].item())
        print_params("X", gt['x'], opt.pos[0].item())


def run_donut_chart_test():
    print("\n" + "="*50)
    print("TEST CASE 2: Donut Chart Optimization")
    print("="*50)
    
    W, H = 256, 256
    
    # ================= 1. 生成目标图 =================
    target_renderer = DiffChartRenderer(W, H)
    
    # 参数：中心(128,128), 内径50, 外径100, 起始0, 扫过 3/4 圆 (4.71 rad)
    gt_cx, gt_cy = 128.0, 128.0
    gt_r_in, gt_r_out = 50.0, 100.0
    gt_start, gt_sweep = 0.0, 4.71
    
    gt_donut = DonutElement(gt_cx, gt_cy, gt_r_in, gt_r_out, gt_start, gt_sweep, 
                            color=torch.tensor([0.9, 0.6, 0.2, 1.0]))
    target_renderer.add_element(gt_donut)
    
    target_image = target_renderer()
    save_image(target_image, "donut_target.png")
    
    # ================= 2. 初始化待优化模型 =================
    opt_renderer = DiffChartRenderer(W, H)
    
    # 初始猜测：也是个甜甜圈，但半径不对，角度也不对
    # 初始是个很细的圈，角度很小
    init_donut = DonutElement(128, 128, 70.0, 80.0, 0.0, 1.0, 
                              color=torch.tensor([0.9, 0.6, 0.2, 1.0]))
    
    opt_renderer.add_element(init_donut)
    
    save_image(opt_renderer(), "donut_init.png")
    
    # ================= 3. 优化 =================
    optimizer = optim.Adam(opt_renderer.parameters(), lr=0.1) # 角度和半径的数值范围较小，LR调小一点
    
    print("Starting Optimization...")
    for step in range(201):
        optimizer.zero_grad()
        rendered_image = opt_renderer()
        loss = torch.mean((rendered_image - target_image) ** 2)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")
            save_image(rendered_image, f"donut_step_{step:03d}.png")
            
    # ================= 4. 结果 =================
    print("\nParameter Comparison:")
    print_params("R_in", gt_r_in, init_donut.r_in.item())
    print_params("R_out", gt_r_out, init_donut.r_out.item())
    print_params("Sweep", gt_sweep, init_donut.sweep_angle.item())

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Optimization might be slow on CPU.")
    
    try:
        run_bar_chart_test()
        run_donut_chart_test()
        print(f"\nAll tests completed. Results saved in '{OUTPUT_DIR}' directory.")
    except ImportError as e:
        print("Error: Dependency missing.")
        print(e)
    except Exception as e:
        print("An error occurred during execution:")
        print(e)
