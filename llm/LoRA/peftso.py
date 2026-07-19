import json
import torch
from torch.utils.data import Dataset

class QwenConversationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 用于定位助手回复的特殊 token 序列
        self.assistant_prefix_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_end_ids = tokenizer.encode(
            "<|im_end|>\n", add_special_tokens=False
        )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                system = data.get("system", "")
                conversations = data["conversation"]

                # 构建消息列表
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                for turn in conversations:
                    messages.append({"role": "user", "content": turn["human"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

                # 生成完整文本并分词（chat_template 已包含特殊 token）
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                encoding = tokenizer(text, add_special_tokens=False)
                input_ids = encoding["input_ids"]

                # 创建 labels，仅对助手回复计算损失
                labels = self._create_labels(input_ids)

                # 截断到 max_length
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]

                self.samples.append({"input_ids": input_ids, "labels": labels})

    def _find_sub_list(self, sub, seq):
        """返回子序列在主序列中所有出现位置的起始索引"""
        results = []
        s_len = len(sub)
        for i in range(len(seq) - s_len + 1):
            if seq[i:i + s_len] == sub:
                results.append(i)
        return results

    def _create_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        prefix_len = len(self.assistant_prefix_ids)
        end_len = len(self.assistant_end_ids)

        # 找到所有 <|im_start|>assistant\n 的起始位置
        starts = self._find_sub_list(self.assistant_prefix_ids, input_ids)
        for start in starts:
            content_start = start + prefix_len          # 助手内容开始位置
            # 搜索后续的 <|im_end|>\n
            rest = input_ids[content_start:]
            end_pos_in_rest = self._find_sub_list(self.assistant_end_ids, rest)
            if end_pos_in_rest:
                content_end = content_start + end_pos_in_rest[0]   # 助手内容结束位置（不包含结束符）
            else:
                content_end = len(input_ids)            # 截断情况，直到序列末尾
            # 将助手内容区域的 labels 设为原始 token id
            for i in range(content_start, content_end):
                if i < len(labels):
                    labels[i] = input_ids[i]
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------- 1. 加载基座模型 --------------------
model_name = "D:\PycharmWork\demo\llm\localdeploy\Qwen\Qwen2.5-0.5B-Instruct"   # 请修正为你的实际路径，使用正斜杠
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确保 pad_token 存在（Qwen 通常没有显式设置）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- 2. 配置 LoRA --------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,        # 秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],    # 注意力模块 qv
    lora_dropout=0.1,   # 防止过拟合
    bias="none",    # 无偏置
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------- 3. 加载数据集 --------------------
jsonl_path = "./datasets/沐雪_train.jsonl"
train_dataset = QwenConversationDataset(jsonl_path, tokenizer, max_length=512)

# -------------------- 4. 训练配置 --------------------
training_args = TrainingArguments(
    output_dir="./lora_output",     # 输出路径
    per_device_train_batch_size=2,  # 每设备的训练批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数（相当于总批次大小为 8）
    learning_rate=2e-4,             # 学习率
    num_train_epochs=3,             # 训练轮数
    logging_steps=10,               # 日志记录频率
    save_strategy="epoch",          # 每轮结束时保存模型
    fp16=True,                      # 使用混合精度训练（如果你的 GPU 支持）
    dataloader_pin_memory=False,    # 关闭 pin_memory 以避免潜在的内存问题
    remove_unused_columns=False,    # 保持数据集中的所有列，以便 DataCollator 正确处理 labels
)

# DataCollator 只需做动态 padding，labels 已在数据集中准备
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
trainer.train()

# -------------------- 5. 保存 --------------------
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")