#真人编写，部分使用AI
import os
import time
import torch
import requests
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import math

class MiniStarConfig:
    def __init__(self):
        self.vocab_size = 4096
        self.n_embd = 32
        self.n_head = 2
        self.n_layer = 2
        self.block_size = 256
        self.dropout = 0.1  # 这里要注意：与 train_ministar.py 不同，但必须和 chat_ministar.py 一致！
        self.pad_token_id = 0

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = torch.nn.functional.dropout(att, self.dropout, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniStar(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = torch.nn.LayerNorm(config.n_embd)
        self.head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

def generate_ministar_output(model, tokenizer, user_input, device, max_new_tokens=150):
    model.eval()
    prompt = f"指令：{user_input}\n输出："
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated_ids = input_ids.clone()
    
    output_tokens = []
    last_tokens = []
    punctuation_run = 0
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated_ids)
            next_token_logits = logits[:, -1, :] / 0.8 
            
            for token_id in last_tokens[-8:]:
                next_token_logits[0, token_id] -= 1.5
                
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
            
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        if decoded in "：。（）【】、，；！？":
            punctuation_run += 1
        else:
            punctuation_run = 0
        if punctuation_run > 6:
            break
            
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        last_tokens.append(token_id)
        
        if token_id not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
            output_tokens.append(token_id)
    return tokenizer.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def polish_output_with_gpt5_nano(user_input, raw_output):
    url = "https://api.jkyai.top/API/gpt5-nano"#在这里，特别感谢云智api提供的公益接口!
    system_prompt = (
        "你是一个语言润色专家，你专注于教会小模型生成自然，连续的句子。请将以下由小型AI模型生成的回复，在不改变其核心意思、语气和事实的前提下，"
        "改写成更流利、自然、符合中文母语者表达习惯的句子。如果小模型输出过于错误或重复的句子，请直接全部改写。仅输出改写后的文本，不要包含任何额外说明。"
    )
    question = f"用户问：{user_input}\nAI原始回复：{raw_output}"
    try:
        response = requests.get(
            url,
            params={"question": question, "system": system_prompt},
            timeout=20,
            headers={"User-Agent": "ministar-trainer"}
        )
        if response.status_code == 200:
            polished = response.text.strip()
            return polished if polished else raw_output
    except Exception as e:
        print(f"API 异常: {e}")
    return raw_output

class InteractiveDataset(Dataset):
    def __init__(self, samples, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []
        for user_input, polished_output in samples:
            full_text = f"指令：{user_input}\n输出：{polished_output}{tokenizer.eos_token}"
            tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            if len(tokens) > 5:
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        pad_length = self.block_size - len(tokens)
        if pad_length > 0:
            tokens = torch.cat([tokens, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
        else:
            tokens = tokens[:self.block_size]
        return tokens

def main():
    if not os.path.exists("ministar.pth"):
        print(" 找不到 ministar.pth")
        return
    if not os.path.exists("ministar_tokenizer.json"):
        print("找不到 ministar_tokenizer.json")
        return

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="ministar_tokenizer.json",
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )

    config = MiniStarConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model = MiniStar(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f" 使用设备: {device}")
    model.load_state_dict(torch.load("ministar.pth", map_location="cpu"))
    model.to(device)
    print("ministar 模型加载成功")
    collected_samples = []
    print("\n 交互式训练")
    print(" 输入 'STOP' 结束并微调")
    while True:
        try:
            user_input = input("\n 用户输入: ").strip()
        except EOFError:
            break
        if user_input.upper() == "STOP":
            break
        if not user_input:
            continue

        raw_output = generate_ministar_output(model, tokenizer, user_input, device)
        print(f"ministar: {raw_output}")

        polished_output = polish_output_with_gpt5_nano(user_input, raw_output)
        print(f"润色后: {polished_output}")
        if polished_output and polished_output != raw_output:
            collected_samples.append((user_input, polished_output))
            print(" 样本已收录")
        else:
            print(" 跳过无效样本")
        
        time.sleep(1)

    if not collected_samples:
        print("无有效样本，退出。")
        return

    print(f"\n 收集 {len(collected_samples)} 条样本，开始微调...")
    dataset = InteractiveDataset(collected_samples, tokenizer, block_size=config.block_size)
    if len(dataset) == 0:
        print("数据集为空")
        return

    dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    total_loss = 0
    for batch in tqdm(dataloader, desc=" 微调中"):
        batch = batch.to(device)
        logits = model(batch[:, :-1])[0]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            ignore_index=config.pad_token_id
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"微调完成！平均 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ministar_newtrain.pth")
    print("模型已保存为 ministar_newtrain.pth")
    print("现在可以运行 chat_ministar.py(注:保存的模型为ministar_newtrain.pth,请手动更改代码!)")

if __name__ == "__main__":
    main()