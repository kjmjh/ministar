<div align="center">
<img src="./ministar.svg"height="200px"width="200px"></img>
<h1>Ministar</h1>
<p>一个超轻量级AI模型+从零开始自己训练!🚀</p>
</div>

> [!NOTE]
> 本教程意在用浅显易懂的语言教使用者快速打造自己的AI模型，如有不对请指出

## 快速使用

1.环境准备

安装Python

```
请自己解决
```

创建虚拟环境(可选但推荐)

```
#创建虚拟环境
python -m venv myenv

#激活虚拟环境
windows: myenv\Scripts\activate
linux: source myenv/bin/activate
```

创建一个文件夹(什么名字都可以，cd 进入)

然后克隆仓库

```
git clone https://github.com/kjmjh/ministar.git
(注:由于模型很小，提前放好了默认模型，有其他需要请去Releases里下载)
```

安装需要的库

```
pip install -r requirements.txt
```

2.运行对话(注:要把所有文件都放在同一个目录!代码是这样写的)

```
python chat_ministar.py

或使用web对话:
streamlit run web_chat.py
```

🎉恭喜你成功运行ministar!

> 移动设备使用正在探索中...

## 接下来:从零开始自己训练

### 前言

AI大模型时代，自己打造一个轻量AI模型已经成为现实。接下来，以本人硬件速度来说，可以让你<30分钟打造自己的AI模型(好不好用不好说，但“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”——minimind)

1.环境准备

```
pip install -r requirements.txt
```

2.准备数据集

这里，我是用的是ShareGPT中的一个子集:computer_zh_26k.jsonl
并转换为alpaca格式，提取其前2000条对话，其格式大概是这样的:

```
{"instruction": "你是谁？", "input": "", "output": "我是ministar，是一个小型语言模型。我可以帮助提供信息，回答问题和生成文本。今天我能为您做些什么？"}
```

当然你也可以准备自己的数据集!
如果你需要我使用的数据集，请在issues里反馈。

3.开始训练

```
git clone https://github.com/kjmjh/ministar.git

(其实你之前已经git过了)
```

```
python train_ministar.py
#注:需要把你的数据集重命名为computer_zh_2k_alpaca.jsonl，tokenizer默认保存为ministar_tokenizer.json
你可以自行在代码里更改名称，大约在132,164,168行(train_ministar.py)。
模型权重默认保存为ministar.pth,你可以在大约186,188,212,213行(train_ministar.py)里根据你的实际情况更改。并且在chat_ministar里也要改为一样的!
```

> 其它特别参数请查看代码，那里我尽量写了详细注释

4.运行AI模型

```
python chat_ministar.py
#确保模型权重文件，tokenizer等等都在同一个目录下，这很重要!

或启动web对话: streamlit run web_chat.py
```

5.*(可选):可以运行

```
python safe_ministar.py
#代码里默认寻找的文件名为"ministar.pth"，保存为"model.safetensors",请根据你的实际情况修改代码，或者原封不动
```

将模型权重文件转换为safetensors格式

> 至于为什么这样做，请自行搜索

🎉恭喜你，成功训练了一个轻量AI模型!

## 常见问题解答

> 敬请期待

## 开源协议

本项目使用GPL-2.0开源协议。你可以自由使用本项目，并进行二次分发。但你的分发项目也必须开源并使用和本项目相同的开源协议。

这意味着本项目限制商用!

请遵守，谢谢。

> *作者还只是一个初中生呢，能不能点个star支持一下ヽ(*⌒∇⌒*)ﾉ*