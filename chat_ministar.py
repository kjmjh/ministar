#真人编写，非AI
import os
import torch
from transformers import PreTrainedTokenizerFast

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

#模型配置（注:必须与训练时一致，在使用前请对照train_ministar.py:))(())
class MiniStarConfig:
    def __init__(self):
        self.vocab_size = 4096  #一定一定不要错了!
        self.n_embd = 32
        self.n_head = 2
        self.n_layer = 2
        self.block_size = 256
        self.dropout = 0.1
        self.pad_token_id = 0

import math
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

def main(): #以下文件名根据你的实际情况决定
    model_path_pth = "ministar.pth"
    model_path_safetensors = "model.safetensors"
    tokenizer_path = "ministar_tokenizer.json"
    
    model_path = None
    if os.path.exists(model_path_pth):
        model_path = model_path_pth
    elif os.path.exists(model_path_safetensors):
        model_path = model_path_safetensors
    else:
        print(f"错误: 找不到模型文件 '{model_path_pth}' 或 '{model_path_safetensors}'")
        print("请先运行: python train_ministar.py")
        return

    if not os.path.exists(tokenizer_path):
        print(f"错误: 找不到 tokenizer '{tokenizer_path}'")
        return

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]"
    )
    
    config = MiniStarConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model = MiniStar(config)
    
    if model_path.endswith(".safetensors"):
        if not SAFETENSORS_AVAILABLE:
            print("错误: 需要安装 safetensors 库以加载 .safetensors 文件，这很重要!")
            print("运行: pip install safetensors")
            return
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    print("输入 'q' 退出。")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n 你: ").strip()
            if user_input.lower() == "q":
                print(" 再见！")
                break
            if not user_input:
                continue

            prompt = f"指令：{user_input}\n输出："
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            generated_ids = input_ids.clone()

            print(" ministar: ", end="", flush=True)
            new_text_cache = ""  
        #好鸡肋
            last_tokens = [] 
            punctuation_run = 0 
            for _ in range(150):  
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
                if decoded in "：。（）【】、，；！？":#用于连续标点检测
                    punctuation_run += 1
                else:
                    punctuation_run = 0
                if punctuation_run > 6:  #这里的说明:改进train_ministar.py后，AI更加胡言乱语，所以防止AI生成连续6个以上的字符，并打断
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                last_tokens.append(token_id)

                if token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id]:
                    continue

                new_token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                print(new_token_text, end="", flush=True)

            print() 

        except KeyboardInterrupt:
            print("\n 再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            break

if __name__ == "__main__":
    main()