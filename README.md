# iNeuOS_LLM
训练自主的领域大模型
基于DeepSeek-R1-Distill-Qwen-1.5B模型训练自主的领域大模型，例如打造针对钢铁、矿山、有色、稀土、建材、加工制造等领域的大模型。促进大模型技术在具体领域的应用与落地，助力行业向智能制造转型升级。

# 微调模型训练过程及代码
## 1.代码工程目录
<img width="1005" height="688" alt="image" src="https://github.com/user-attachments/assets/df5f859e-ee57-49a4-ac53-e4a884bb2691" />

## 2.下载基础模型DeepSeek-R1-Distill-Qwen-1.5B
使用huggingface-cli下载基础模型,工具下载地址：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## 3.准备数据集
（1）	把准备好的Word文件，放入docx_input文件夹。
（2）	运行word_to_dataset.py代码，会生成training_dataset.json数据集文件，把training_dataset.json文件名称改为datasets.json。数据集有很大的调优空间。

## 4.训练模型
（1）	运行train_model.py，训练模型，以我的笔记本配置，训练完成datasets.json数据集需要将近4个小时。
（2）	训练完成的模型保存在deepseek-1.5b-lora-final目录下。
（3）	使用test_train_model.py代码测试基础模型与训练后的模型。

## 5.合并模型
（1）	运行merged_model.py代码，把基础模型与训练后的模型合并成一个整体的模型，保存在deepseek-1.5b-lora-merged目录下。
（2）	运行test_merged_model.py代码，测试合并后的模型。应用效果参见本文章节：测试微调训练后的大模型。

# iNeuOS工业互联网公众号：

![iNeuOS工业互联网公众号](https://img2020.cnblogs.com/blog/279374/202011/279374-20201109210223158-1810580141.jpg)

# 官方网站:http://www.ineuos.net<br>
