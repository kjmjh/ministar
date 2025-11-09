# AI编写，非真人
import streamlit as st
import torch
from transformers import PreTrainedTokenizerFast
import os
import sys
from typing import Generator
import time

sys.path.append(os.path.dirname(__file__))

try:
    from chat_ministar import MiniStar, MiniStarConfig
except ImportError as e:
    st.error(f"无法导入模型: {e}")
    st.stop()

# 模型路径配置（与 chat_ministar.py 一致）
MODEL_PATH_PTH = "ministar_finetuned.pth"
MODEL_PATH_SAFETENSORS = "model.safetensors"
TOKENIZER_PATH = "ministar_tokenizer.json"

@st.cache_resource
def load_model_and_tokenizer():
    model_path = None
    if os.path.exists(MODEL_PATH_PTH):
        model_path = MODEL_PATH_PTH
    elif os.path.exists(MODEL_PATH_SAFETENSORS):
        model_path = MODEL_PATH_SAFETENSORS
    else:
        st.error(f"找不到模型文件！请确保存在 '{MODEL_PATH_PTH}' 或 '{MODEL_PATH_SAFETENSORS}'，以及 '{TOKENIZER_PATH}'。")
        st.stop()

    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"找不到 tokenizer 文件：{TOKENIZER_PATH}")
        st.stop()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]"
    )
    
    config = MiniStarConfig()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model = MiniStar(config)

    # 加载权重
    if model_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        except ImportError:
            st.error("需要安装 safetensors 来加载 .safetensors 文件：pip install safetensors")
            st.stop()
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

def generate_stream(model, tokenizer, device, prompt: str, max_new_tokens: int = 150) -> Generator[str, None, None]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated_ids = input_ids.clone()

    last_tokens = []
    punctuation_run = 0
    punctuation_set = set("：。（）【】、，；！？")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated_ids)
            next_token_logits = logits[:, -1, :] / 0.8  # temperature = 0.8

            # 重复惩罚：对最近 8 个 token 降低概率
            for token_id in last_tokens[-8:]:
                next_token_logits[0, token_id] -= 1.5

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()

        # 检查 EOS
        if token_id == tokenizer.eos_token_id:
            break

        # 解码当前 token（跳过特殊 token）
        if token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id]:
            last_tokens.append(token_id)
            continue

        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        
        # 标点连续检测
        if decoded in punctuation_set:
            punctuation_run += 1
        else:
            punctuation_run = 0

        if punctuation_run > 6:  # 防止胡言乱语
            break

        # 更新状态
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        last_tokens.append(token_id)

        # 输出文本
        if decoded:
            yield decoded

# Streamlit 界面
def main():
    st.set_page_config(page_title="ministar Chat", page_icon="⭐", layout="wide")
    st.title(" ministar AI")
    st.caption("一个胡言乱语的AI模型，by:kjmjh")

    # === 关键：注入CSS以实现用户消息在右，AI消息在左 ===
    # 此方案是社区公认的有效方法 [[3], [4], [9]]
    st.markdown("""
    <style>
    /* 针对用户消息：将其容器内容反转，实现靠右 */
    [data-testid="chat-message"] > div > div > div:nth-child(2) {
        flex-direction: row-reverse;
        text-align: right;
    }
    /* 确保用户消息的文字也靠右 */
    [data-testid="chat-message"] > div > div > div:nth-child(2) > div {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

    model, tokenizer, device = load_model_and_tokenizer()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            # 使用自定义头像
            with st.chat_message("assistant", avatar="ministar.svg"):
                st.markdown(msg["content"])

    # 处理用户新输入
    if prompt := st.chat_input("给 ministar 发送消息……"):
        # 添加并显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        full_prompt = f"指令：{prompt}\n输出："
        full_response = ""
        total_tokens = 0
        start_time = time.time()

        with st.chat_message("assistant", avatar="ministar.svg"):
            message_placeholder = st.empty()
            # 流式生成回复
            for new_text in generate_stream(model, tokenizer, device, full_prompt):
                full_response += new_text
                total_tokens += 1
                message_placeholder.markdown(full_response + "▌")  # 闪烁光标

            # 计算生成速度
            end_time = time.time()
            elapsed_time = end_time - start_time
            speed = total_tokens / elapsed_time if elapsed_time > 0 else 0

            # 将速度信息追加到回复末尾
            final_response = full_response + f"\n\n---\n**生成速度**: {speed:.2f} token/s"
            message_placeholder.markdown(final_response)
            # 仅将纯文本回复存入历史记录
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()