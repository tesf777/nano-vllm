import os
import glob
import torch
import torch.distributed as dist
from transformers import Qwen3Config, AutoConfig
from nanovllm.models import qwen3


def init_torch_distributed():
    """初始化torch分布式环境"""
    if not dist.is_initialized():
        # 设置环境变量
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # 初始化分布式进程组
        dist.init_process_group(
            backend="gloo",  # 使用gloo后端，避免CUDA依赖
            rank=0,
            world_size=1
        )


def print_model_structure(model):
    """打印模型结构、参数信息和torch size"""
    print("\n" + "="*80)
    print("模型结构详细信息")
    print("="*80)

    def print_module_info(module, prefix=""):
        """递归打印模块信息"""
        for name, child in module.named_children():
            module_name = f"{prefix}.{name}" if prefix else name
            print(f"\n📦 模块: {module_name}")
            print(f"   类型: {type(child).__name__}")

            # 打印参数信息
            param_count = 0
            for param_name, param in child.named_parameters():
                if param.data is not None:
                    param_count += 1
                    param_size = param.data.numel()
                    param_shape = tuple(param.data.shape)
                    print(f"   参数 {param_name}: {param_shape}, 元素数量: {param_size:,}, "
                          f"大小: {param_size * param.data.element_size() / 1024 / 1024:.2f} MB")

            print(f"   参数总数: {param_count}")

            # 如果有子模块，递归打印
            if len(list(child.children())) > 0:
                print_module_info(child, module_name)

    print_module_info(model)

    # 打印总参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*80)
    print("模型总统计")
    print("="*80)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")


def print_weight_details(model_path):
    """直接打印safetensors文件中的权重信息"""
    print("\n" + "="*80)
    print("权重文件详细信息")
    print("="*80)

    from safetensors import safe_open

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    print(f"找到 {len(safetensor_files)} 个safetensors文件:")

    weight_info = []
    total_size = 0

    for file_path in safetensor_files:
        file_name = os.path.basename(file_path)
        print(f"\n📄 文件: {file_name}")

        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                weight_names = list(f.keys())
                print(f"   权重数量: {len(weight_names)}")

                file_total_size = 0
                for weight_name in sorted(weight_names):
                    try:
                        tensor = f.get_tensor(weight_name)
                        tensor_size = tensor.numel()
                        tensor_shape = tuple(tensor.shape)
                        memory_size = tensor_size * tensor.element_size() / 1024 / 1024

                        # 只收集weight_name和torchsize信息
                        weight_info.append({
                            'name': weight_name,
                            'shape': tensor_shape,
                            'size': tensor_size,
                            'memory_mb': memory_size
                        })

                        file_total_size += memory_size
                    except Exception as e:
                        print(f"   ❌ {weight_name}: 读取错误 - {e}")

                print(f"   文件总大小: {file_total_size:.2f} MB")
                total_size += file_total_size

        except Exception as e:
            print(f"   ❌ 文件读取错误: {e}")

    print(f"\n所有权重文件总大小: {total_size:.2f} MB")
    return weight_info


def save_to_markdown(model_path, weight_info, total_params=None):
    """将权重信息保存到Markdown文件"""
    # 生成输出文件名
    model_name = os.path.basename(model_path.rstrip('/'))
    output_file = os.path.join(model_path, f"{model_name}_weights_info.md")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} 模型权重信息\n\n")

        # 基本信息
        f.write("## 基本信息\n\n")
        f.write(f"- **模型路径**: `{model_path}`\n")
        f.write(f"- **权重数量**: {len(weight_info)}\n")
        if total_params:
            f.write(f"- **总参数数量**: {total_params:,}\n")

        # 计算总内存
        total_memory = sum(w['memory_mb'] for w in weight_info)
        f.write(f"- **总内存占用**: {total_memory:.2f} MB\n\n")

        # 权重详情表格
        f.write("## 权重详情\n\n")
        f.write("| 权重名称 | 形状 | 元素数量 | 内存 (MB) |\n")
        f.write("|---------|------|----------|----------|\n")

        for weight in weight_info:
            f.write(f"| {weight['name']} | {weight['shape']} | {weight['size']:,} | {weight['memory_mb']:.2f} |\n")

        # 按内存大小排序的权重
        f.write("\n## 按内存大小排序的权重 (Top 20)\n\n")
        f.write("| 权重名称 | 形状 | 内存 (MB) |\n")
        f.write("|---------|------|----------|\n")

        sorted_weights = sorted(weight_info, key=lambda x: x['memory_mb'], reverse=True)[:20]
        for weight in sorted_weights:
            f.write(f"| {weight['name']} | {weight['shape']} | {weight['memory_mb']:.2f} |\n")

    print(f"\n✅ 权重信息已保存到: {output_file}")
    return output_file


if __name__ == "__main__":
    # 示例：使用 load_model 函数读取 ./Qwen3-0.6B 路径下的模型
    model_path = "./Qwen3-0.6B"

    # 检查模型路径是否存在
    if os.path.exists(model_path):
        print(f"正在分析模型从路径: {model_path}")

        try:
            # 读取权重信息并保存到Markdown文件
            weight_info = print_weight_details(model_path)
            total_params = None

            # 可选：加载模型获取总参数数
            try:
                print("\n🔄 正在加载模型以获取总参数数...")
                # 初始化分布式环境
                init_torch_distributed()

                # 加载模型配置
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    config = AutoConfig.from_pretrained(model_path)
                    print(f"✅ 成功加载配置")
                else:
                    # 使用默认配置
                    config = Qwen3Config(
                        vocab_size=152064,
                        hidden_size=896,
                        intermediate_size=4864,
                        num_hidden_layers=24,
                        num_attention_heads=14,
                        num_key_value_heads=2,
                        max_position_embeddings=32768,
                        rms_norm_eps=1e-6,
                        hidden_act="silu"
                    )
                    print(f"⚠️  配置文件不存在，使用默认配置")

                # 创建模型实例
                model = qwen3.Qwen3ForCausalLM(config)
                print(f"✅ 成功创建模型实例")

                # 计算总参数数
                total_params = sum(p.numel() for p in model.parameters())
                print(f"✅ 模型总参数数: {total_params:,}")

            except Exception as e:
                print(f"⚠️  无法加载模型获取总参数数: {e}")
                print("   将继续生成权重信息报告（不包含总参数数）")

            finally:
                # 清理分布式环境
                if dist.is_initialized():
                    dist.destroy_process_group()

            # 保存到Markdown文件
            save_to_markdown(model_path, weight_info, total_params)

        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"❌ 模型路径不存在: {model_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"请检查模型路径是否正确")