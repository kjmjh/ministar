#AI编写，非真人
import streamlit as st
import torch
from transformers import PreTrainedTokenizerFast
import os
import sys
from typing import Generator

sys.path.append(os.path.dirname(__file__))

try:
    from chat_ministar import MiniStar, MiniStarConfig
except ImportError as e:
    st.error(f"无法导入模型: {e}")
    st.stop()

# 初始化 
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "ministar.pth"
    tokenizer_path = "ministar_tokenizer.json"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error(" 找不到模型文件!请确保ministar.pth，ministar_tokenizer.json，chat_ministar.py都在同一个目录下!")
        st.stop()

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
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

def generate_stream(model, tokenizer, device, prompt: str, max_new_tokens: int = 120) -> Generator[str, None, None]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated_ids = input_ids.clone()
    
    new_text_cache = ""
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated_ids)
            next_token_logits = logits[:, -1, :] / 0.5  
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        new_token_id = next_token.item()
        
        if new_token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id]:
            continue
            
        new_token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
        
        if new_token_text.startswith("Ġ"):
            if new_text_cache:
                yield new_text_cache
                new_text_cache = ""
            yield new_token_text[1:]
        else:
            new_text_cache += new_token_text
    
    if new_text_cache:
        yield new_text_cache

#Streamlit 界面
def main():
    st.set_page_config(page_title="ministar Chat", page_icon="M⭐", layout="wide")
    st.title(" ministar AI")
    st.caption("一个胡言乱语的AI模型，by:kjmjh")

    model, tokenizer, device = load_model_and_tokenizer()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("给ministar发送消息......"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        full_prompt = f"指令：{prompt}\n输出："

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for new_text in generate_stream(model, tokenizer, device, full_prompt):
                full_response += new_text
                message_placeholder.markdown(full_response + "I")  
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()