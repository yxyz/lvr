"""
测试 BLINK 数据集并计算正确率
支持多个数据集配置：Counting, IQ_Test 等
Test BLINK datasets and calculate accuracy
Supports multiple dataset configs: Counting, IQ_Test, etc.
"""
import os

# 在导入之前设置环境变量，确保 Triton 能找到 C 编译器
if "CC" not in os.environ:
    os.environ["CC"] = "gcc"
if "CXX" not in os.environ:
    os.environ["CXX"] = "g++"

os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import torch
import json
from datasets import load_dataset
import string
from tqdm import tqdm
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
    
    # 准备LVR参数
    lvr_steps_list = [lvr_steps] if decoding_strategy == "steps" else None
    
    # 生成
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


def accuracy_reward(response: str, ground_truth: str) -> bool:
    """计算答案是否正确"""
    # 从响应中提取答案
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer.upper() == ground_truth.upper()


def process_blink_data(dat):
    """处理 BLINK 数据集中的数据项"""
    idx = dat["idx"]
    choices = dat["choices"]
    letters = string.ascii_uppercase 
    paired = list(zip(letters, choices))
    option_string = ""
    for letter, choice in paired:
        option_string += f"{letter}. {choice}\n"
    
    # 提取答案
    if len(dat['answer']) > 1:
        ans = dat['answer'][1].upper()
    else:
        ans = dat['answer'][0].upper()
    
    # 提取图片
    images = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in dat and dat[k] is not None:
            images.append(dat[k])
    
    # 构建问题
    question = dat['question'] + "\nOptions:\n" + option_string
    task_instruction = "\nAnswer with the option's letter from the given choices directly."
    full_question = question + task_instruction
    
    return {
        "question_id": idx,
        "image": images,
        "query": full_question,
        "label": ans,
    }


def test_dataset(
    model,
    processor,
    dataset_config,
    decoding_strategy="steps",
    lvr_steps=8,
    max_new_tokens=512
):
    """
    测试单个数据集
    
    Args:
        model: 加载的模型
        processor: 处理器
        dataset_config: 数据集配置名称（如 "Counting", "IQ_Test"）
        decoding_strategy: 解码策略
        lvr_steps: LVR步数
        max_new_tokens: 最大生成token数
    
    Returns:
        dict: 包含测试结果的字典
    """
    print(f"\n{'='*80}")
    print(f"测试数据集: {dataset_config}")
    print("="*80)
    
    # 加载数据集
    print(f"\n正在加载 BLINK {dataset_config} 数据集...")
    try:
        ds = load_dataset("BLINK-Benchmark/BLINK", dataset_config)
        val_dataset = ds['val']
        print(f"数据集加载完成，共有 {len(val_dataset)} 个测试样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保已登录 Hugging Face: huggingface-cli login")
        return None
    
    # 测试
    print("\n开始测试...")
    total = 0
    correct = 0
    results = []
    
    for dat in tqdm(val_dataset, desc=f"测试 {dataset_config}"):
        # 处理数据
        processed = process_blink_data(dat)
        
        # 运行推理
        try:
            outputs = run_inference(
                model=model,
                processor=processor,
                img_path=processed["image"],
                text=processed["query"],
                decoding_strategy=decoding_strategy,
                lvr_steps=lvr_steps,
                max_new_tokens=max_new_tokens
            )
            
            prediction = outputs[0]
            label = processed["label"]
            
            # 计算是否正确
            is_correct = accuracy_reward(prediction, label)
            if is_correct:
                correct += 1
            total += 1
            
            # 保存结果
            results.append({
                "question_id": processed["question_id"],
                "prediction": prediction,
                "label": label,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"\n处理样本 {processed['question_id']} 时出错: {e}")
            continue
    
    # 计算准确率
    accuracy = correct/total*100 if total > 0 else 0
    
    # 输出结果
    print("\n" + "="*80)
    print(f"测试结果 - {dataset_config}")
    print("="*80)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"错误数: {total - correct}")
    print(f"正确率: {accuracy:.2f}%")
    print("="*80)
    
    return {
        "dataset_config": dataset_config,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def main():
    # ========== 配置参数 - 请在此处修改 ==========
    checkpoint_path = "LVR-7B"  # 模型checkpoint路径
    decoding_strategy = "steps"  # 解码策略: "steps" 或 "latent"
    lvr_steps = 8  # LVR推理步数（当decoding_strategy="steps"时使用）
    max_new_tokens = 512  # 最大生成token数
    use_flash_attention = False  # 是否使用Flash Attention 2
    
    # 要测试的数据集配置列表
    # 支持的配置: Counting, IQ_Test, Jigsaw, Relative_Reflectance, Spatial_Relation 等
    dataset_configs = ["Counting", "IQ_Test","Jigsaw","Relative_Reflectance","Spatial_Relation"]  # 想要测哪些数据集就添加到这个列表中
    # ==========================================
    
    print("="*80)
    print("BLINK 数据集测试")
    print("="*80)
    print(f"模型路径: {checkpoint_path}")
    print(f"数据集配置: {', '.join(dataset_configs)}")
    print(f"解码策略: {decoding_strategy}")
    print(f"LVR步数: {lvr_steps}")
    print("="*80)
    
    # 创建结果保存文件夹
    result_dir = "val_result"
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n结果将保存到: {result_dir}/")
    
    # 加载模型（只加载一次，供所有数据集使用）
    print("\n正在加载模型...")
    model, processor = load_model_and_processor(
        checkpoint_path, 
        use_flash_attention=use_flash_attention
    )
    
    # 测试所有指定的数据集
    all_results = []
    summary_results = []
    
    for dataset_config in dataset_configs:
        result = test_dataset(
            model=model,
            processor=processor,
            dataset_config=dataset_config,
            decoding_strategy=decoding_strategy,
            lvr_steps=lvr_steps,
            max_new_tokens=max_new_tokens
        )
        
        if result is not None:
            all_results.append(result)
            summary_results.append({
                "dataset": dataset_config,
                "total": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"]
            })
            
            # 为每个数据集保存单独的结果文件
            output_file = os.path.join(result_dir, f"blink_{dataset_config.lower()}_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到: {output_file}")
    
    # 输出汇总结果
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("所有数据集测试结果汇总")
        print("="*80)
        total_all = sum(r["total"] for r in all_results)
        correct_all = sum(r["correct"] for r in all_results)
        accuracy_all = correct_all/total_all*100 if total_all > 0 else 0
        
        print(f"\n各数据集结果:")
        for summary in summary_results:
            print(f"  {summary['dataset']:20s}: {summary['correct']:4d}/{summary['total']:4d} = {summary['accuracy']:6.2f}%")
        
        print(f"\n总体结果:")
        print(f"  总样本数: {total_all}")
        print(f"  总正确数: {correct_all}")
        print(f"  总体正确率: {accuracy_all:.2f}%")
        print("="*80)
        
        # 保存汇总结果
        summary_file = os.path.join(result_dir, "blink_all_results_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary_results,
                "overall": {
                    "total": total_all,
                    "correct": correct_all,
                    "accuracy": accuracy_all
                },
                "detailed_results": all_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    main()

