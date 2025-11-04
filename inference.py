"""
简单的推理脚本示例
Simple inference script for LVR model

使用方法 (Usage):
    直接在代码中修改下面的配置参数，然后运行: python inference.py
"""
import os

# 在导入之前设置环境变量，确保 Triton 能找到 C 编译器
# 这对于 Triton 编译 CUDA 工具是必需的
if "CC" not in os.environ:
    os.environ["CC"] = "gcc"
if "CXX" not in os.environ:
    os.environ["CXX"] = "g++"

# 尝试禁用一些可能导致问题的 Triton 特性
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import torch
from transformers import AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr


def create_messages(img_path, question):
    """创建消息格式"""
    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = []
        for ip in img_path:
            vision_content.append({
                "type": "image",
                "image": ip,
            })
        vision_content.append({"type": "text", "text": question})
        messages = [
            {
                "role": "user",
                "content": vision_content,
            }
        ]
    return messages


def load_model_and_processor(checkpoint_path, use_flash_attention=False):
    """加载模型和处理器"""
    print(f"正在加载模型从: {checkpoint_path}")
    
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # 替换前向传播函数以支持LVR推理
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True,
        lvr_head=config.lvr_head
    )
    
    # 选择注意力实现方式
    if use_flash_attention:
        attn_impl = "flash_attention_2"
        print("使用 Flash Attention 2")
    else:
        attn_impl = "sdpa"
        print("使用 SDPA (Scaled Dot Product Attention)")
    
    # 加载模型
    model = QwenWithLVR.from_pretrained(
        checkpoint_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=attn_impl,
        device_map="auto",
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    print("模型加载完成!")
    return model, processor


def run_inference(
    model, 
    processor, 
    img_path, 
    text, 
    decoding_strategy="steps",
    lvr_steps=8,
    max_new_tokens=512
):
    """
    运行推理
    
    Args:
        model: 加载的模型
        processor: 处理器
        img_path: 图片路径（可以是单个路径或路径列表）
        text: 问题文本
        decoding_strategy: 解码策略 ("steps" 或 "latent")
        lvr_steps: LVR步数（当decoding_strategy="steps"时使用）
        max_new_tokens: 最大生成token数
    """
    # 创建消息
    messages = create_messages(img_path, text)
    text_formatted = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 准备输入
    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备LVR参数
    lvr_steps_list = [lvr_steps] if decoding_strategy == "steps" else None
    
    # 生成
    print(f"正在推理 (解码策略: {decoding_strategy}, LVR步数: {lvr_steps})...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            lvr_steps=lvr_steps_list
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
    
    return output_text


def main():
    # ========== 配置参数 - 请在此处修改 ==========
    checkpoint_path = "LVR-7B"  # 模型checkpoint路径
    image_path = "test.png"  # 图片路径（可以是单个图片路径字符串，或图片路径列表）
    question = "how many mountains are in the image?"  # 问题文本
    decoding_strategy = "steps"  # 解码策略: "steps" 或 "latent"
    lvr_steps = 8  # LVR推理步数（当decoding_strategy="steps"时使用）
    max_new_tokens = 512  # 最大生成token数
    use_flash_attention = False  # 是否使用Flash Attention 2（需要C编译器，如果报错请设为False使用sdpa）
    # ==========================================
    
    # 处理图片路径（支持多张图片）
    if isinstance(image_path, str) and ',' in image_path:
        img_paths = [p.strip() for p in image_path.split(',')]
    elif isinstance(image_path, list):
        img_paths = image_path
    else:
        img_paths = image_path
    
    # 加载模型
    model, processor = load_model_and_processor(checkpoint_path, use_flash_attention=use_flash_attention)
    
    # 运行推理
    outputs = run_inference(
        model=model,
        processor=processor,
        img_path=img_paths,
        text=question,
        decoding_strategy=decoding_strategy,
        lvr_steps=lvr_steps,
        max_new_tokens=max_new_tokens
    )
    
    # 输出结果
    print("\n" + "="*80)
    print("推理结果 (Inference Result):")
    print("="*80)
    print(outputs[0])
    print("="*80)


if __name__ == "__main__":
    main()
