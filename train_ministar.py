#真人编写，部分使用AI。by:kjmjh
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import math

#模型配置(注意，这里一定一定不要错了!和chat_ministar.py需一致!!!)
class MiniStarConfig:
    def __init__(self):
        self.vocab_size = 4096
        self.n_embd = 32
        self.n_head = 2
        self.n_layer = 2
        self.block_size = 256
        self.dropout = 0.3
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id
            )
            return logits, loss
        return logits, None

#整合了tokenizer训练,更加便捷了。by:kjmjh 2025/10/31
def train_tokenizer(data_file, vocab_size=8192):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    def batch_iterator(batch_size=1000):
        with open(data_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                item = json.loads(line)
                text = item["instruction"] + "\n" + item["output"]
                lines.append(text)
                if len(lines) == batch_size:
                    yield lines
                    lines = []
            if lines: yield lines
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save("ministar_tokenizer.json")

class AlpacaDataset(Dataset):
    def __init__(self, data_file, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)


                prompt = f"指令：{item['instruction']}\n输出："
                completion = item["output"]
                full = prompt + completion + tokenizer.eos_token
                tokens = tokenizer.encode(full, add_special_tokens=False)
                if len(tokens) > block_size:
                    tokens = tokens[:block_size]
                self.samples.append(torch.tensor(tokens, dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        pad_length = self.block_size - len(tokens)
        if pad_length > 0:
            tokens = torch.cat([tokens, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
        else:
            tokens = tokens[:self.block_size]
        return tokens
        
        #这里不会写了，用AI吧:P

def main():
    data_file = "computer_zh_2k_alpaca.jsonl"#这里的文件名根据你的实际情况决定
    if not os.path.exists(data_file):
        print(f"错误: 找不到训练文件 '{data_file}'")
        return
    if not os.path.exists("ministar_tokenizer.json"):#这里也是
        print(" 正在训练tokenizer")
        train_tokenizer(data_file, vocab_size=8192)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="ministar_tokenizer.json",
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )

    config = MiniStarConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model = MiniStar(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}") #这里为了训练更方便，增加了使用设备查看
    model.to(device)

    if os.path.exists("ministar.pth"):
        print("加载已有模型继续训练...")
        model.load_state_dict(torch.load("ministar.pth", map_location=device))

    print("加载数据集...")
    dataset = AlpacaDataset(data_file, tokenizer, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
                                             # ↑ 注意!!!这里一点要根据实际硬件性能决定!因为训练小模型，默认为6
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()

    print(" 开始训练")
    for epoch in range(5): #这里epoch(你也可以理解为训练周期)默认为5,可以根据你的实际情况决定
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/5"): #这里的5也可以改一下，改成和上面一致即可
            batch = batch.to(device)
            logits, loss = model(batch[:, :-1], batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成,，平均 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ministar.pth")
    print(" 模型已保存为 ministar.pth")
    print(" 训练完成，现在可以运行: python chat_ministar.py")

if __name__ == "__main__":
    main()