# BERT-Jittor
A BERT model built with Jittor.   

# BERT 计图版
本项目采用 计图(Jittor) 框架实现 BERT 模型。

请阅读示例：
[*计图 NLP 教程 - BERT分类器.ipynb*](https://github.com/LetianLee/BERT-Jittor/blob/main/%E8%AE%A1%E5%9B%BE%20NLP%20%E6%95%99%E7%A8%8B%20-%20BERT%E5%88%86%E7%B1%BB%E5%99%A8.ipynb)

## 说明
### 1. 环境与安装
首先，下载并安装 Anaconda3。 官网下载地址为: https://www.anaconda.com/products/distribution#Downloads  
然后，执行以下命令：
```bash
# Create a conda envs
conda create -n bert_jittor python=3.7 ipython
conda activate bert_jittor

# Install Jittor 
pip install jittor

# Install other libraries
pip install jupyter==1.0.0
pip install pandas==1.2.4
pip install matplotlib==3.3.4
pip install seaborn==0.11.1
```

### 2. 启动 Jupyter 并查看示例教程
在命令行中输入 ```jupyter notebook``` 启动。随后即可浏览并运行示例教程。
