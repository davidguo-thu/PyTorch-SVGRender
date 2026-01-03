#!/bin/bash
set -e  # 遇到错误立即退出

# 定义环境名称
ENV_NAME="diffchart_env"

echo "============================================"
echo "   DiffChart Environment Setup Script"
echo "============================================"

# 1. 检查 Conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# 2. 创建 Conda 环境
echo "Step 1: Creating Conda environment '$ENV_NAME'..."
# 使用 Python 3.8 (pydiffvg 对 3.10+ 支持有时有问题，3.8 最稳)
conda create -n $ENV_NAME python=3.8 -y

# 3. 激活环境
echo "Step 2: Activating environment..."
# 脚本中激活 conda 需要特殊处理
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 4. 安装基础依赖
echo "Step 3: Installing PyTorch and basic dependencies..."
# 安装 CPU 版本的 PyTorch (因为检测到你没有 GPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install numpy scikit-image -y
pip install svgwrite svgpathtools cssutils

# 5. 安装 pydiffvg
echo "Step 4: Installing pydiffvg..."

# 尝试安装
echo "   -> Cloning pydiffvg repository..."
if [ -d "diffvg" ]; then
    rm -rf diffvg
fi
git clone https://github.com/BachiLi/diffvg.git
cd diffvg

echo "   -> Patching for CPU-only build..."
# 这是一个关键的 Patch，强制 CMake 不寻找 CUDA
# 我们需要修改 setup.py 或者 CMakeLists.txt，或者通过环境变量控制
# 这里尝试通过环境变量强制使用 CPU
export DIFFVG_CUDA=0

# 更新 git 子模块 (非常重要，包含 pybind11)
git submodule update --init --recursive

echo "   -> Building and installing (this may take a while)..."
# 使用 pip install . 会触发 setup.py
# 注意：如果没有 CUDA，编译可能会产生大量警告，甚至失败
pip install .

cd ..
# 清理
# rm -rf diffvg

echo "============================================"
echo "✅ Setup Completed Successfully!"
echo "============================================"
echo "To start using the library, run:"
echo "    conda activate $ENV_NAME"
echo "    python3 run_optimization_test.py"
