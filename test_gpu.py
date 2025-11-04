#!/usr/bin/env python3
"""
测试 PyTorch GPU 可用性
"""
import torch

def test_pytorch_gpu():
    print("=" * 50)
    print("PyTorch GPU 检测")
    print("=" * 50)
    
    # 基本信息
    print(f"\nPyTorch 版本: {torch.__version__}")
    
    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        # CUDA 版本信息
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        
        # GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"\nGPU 数量: {gpu_count}")
        
        # 每个 GPU 的详细信息
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # 当前使用的 GPU
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f"\n当前 GPU 设备: {current_device}")
        
        # 简单测试：在 GPU 上创建一个张量
        print("\n执行 GPU 测试...")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x @ y  # 矩阵乘法
            print("✓ GPU 计算测试成功！")
            print(f"  测试张量形状: {z.shape}")
            print(f"  测试张量设备: {z.device}")
        except Exception as e:
            print(f"✗ GPU 计算测试失败: {e}")
    else:
        print("\n警告: CUDA 不可用，将使用 CPU")
        print("可能的原因:")
        print("  1. 未安装支持 CUDA 的 PyTorch 版本")
        print("  2. 系统未检测到 NVIDIA GPU")
        print("  3. NVIDIA 驱动程序未正确安装")
        print("  4. CUDA 工具包未正确安装")
    
    print("\n" + "=" * 50)
    return cuda_available

if __name__ == "__main__":
    test_pytorch_gpu()