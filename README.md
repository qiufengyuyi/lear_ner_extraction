# LEAR NER extraction 

部分代码源于https://github.com/qiufengyuyi/sequence_tagging

依赖包：主要是tensorflow==1.15.0，gensim==3.7.3，其余见requirements.txt

目前项目包含针对BERT-MRC优化的LEAR模型，论文参考：https://aclanthology.org/2021.emnlp-main.379.pdf 

针对的数据：目前是基于字符级别标注的实体识别数据。使用网上公开的字符级的中文词向量https://github.com/Embedding/Chinese-Word-Vectors ，请自行下载词向量。

# 不使用词汇增强

## 训练运行脚本：

```shell
bash run_lear.sh
```



## 评测运行脚本：

```shell
python lear_prediction.py
```

# 使用词汇增强

1、运行soft_lexion_processing.py脚本，生成精简后的词向量文件以及词表和字符-词位置信息（先准备好词向量）

2、在run_lear.sh中，增加add_soft_embeddings的选项。

## 训练运行脚本：

```shell
bash run_lear.sh
```

## 评测运行脚本：

lear_prediction.py中的add_soft_embeddings参数选项，default置为True

```shell
python lear_prediction.py
```


