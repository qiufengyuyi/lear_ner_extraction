# -*- coding: utf-8 -*-
import json
# import os
# import configparser
# # import pandas as pd
# import random
import re
import numpy as np
from gensim.models import word2vec, Word2Vec, keyedvectors, KeyedVectors, TfidfModel


def get_word_tf(embedding_file, training_context_datas):
    """
    simple lexicon 对出现在语料中词向量词表中的词，计算样本中每个字符分别在B,M,E,S中的向量表示
    先统计词频和出现的词表，（全量，包括测试集合训练集）
    :param embedding_file:
    :return:
    """
    vector_base = KeyedVectors.load_word2vec_format(embedding_file)
    # vector_base = KeyedVectors.load(embedding_file, mmap='r')
    embedding_vocab_list = vector_base.wv.index2word
    vocab_count_dict = {key: 0 for key in embedding_vocab_list}
    # {char+B:[word1,word2,...]}
    gaz_position_dict = {}
    text_list = []
    # for id,text_ele in training_context_datas.items():
    #     for key,text in text_ele.items():
    #         text_list.append(text)
    embedding_dict = {}
    text_str = "\n".join(training_context_datas)
    words_trim_list = []
    # words_ids_trim_list = []
    for index, vocab_local in enumerate(vocab_count_dict):
        if index % 500 == 0 and index > 0:
            print(index)
            # break
        try:
            match_count = len(re.findall(r"" + vocab_local, text_str))
            if match_count < 5:
                words_trim_list.append(vocab_local)
                continue
            vocab_count_dict[vocab_local] += len(re.findall(r"" + vocab_local, text_str))
            for index, char in enumerate(vocab_local):
                if len(vocab_local) == 1:
                    # single
                    if char in gaz_position_dict:
                        if "S" not in gaz_position_dict[char]:
                            gaz_position_dict[char]["S"] = [vocab_local]
                        else:
                            gaz_position_dict[char]["S"].append(vocab_local)
                    else:
                        gaz_position_dict[char] = {"S": [vocab_local]}
                        # gaz_position_dict[char]["S"] = [vocab_local]
                else:
                    if index == 0:

                        # start
                        if char in gaz_position_dict:
                            if "B" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["B"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["B"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"B": [vocab_local]}
                    elif index < len(vocab_local) - 1:
                        # middle
                        if char in gaz_position_dict:
                            if "M" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["M"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["M"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"M": [vocab_local]}
                    else:
                        # end
                        if char in gaz_position_dict:
                            if "E" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["E"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["E"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"E": [vocab_local]}

            # if vocab_count_dict[vocab_local] < 5:
            #     # id_vocab = vector_base.wv.vocab[vocab_local].index
            #     # del vector_base.wv.vocab[vocab_local]
            #     # vector_base.wv.vectors =

            #     # words_ids_trim_list.append(vector_base.wv.vocab[vocab_local].index)
            #     # embedding_dict[vocab] = np.array(vector_base[vocab]).tolist()
        except Exception as e:
            print(e)
            continue
    # for id,text_ele in training_context_datas.items():
    #     for key,text in text_ele.items():
    #         for vocab in vocab_count_dict:
    #             try:
    #                 vocab_count_dict[vocab] += len(re.findall(r""+vocab,text))
    #             except:
    #                 continue
    # vocab_count_dict = {key:value for key,value in vocab_count_dict.items() if value > 0}
    # for w in words_trim_list:
    #     del vector_base.wv.vocab[w]
    # vector_base.wv.vectors = np.delete(vector_base.wv.vectors, words_ids_trim_list, axis=0)
    # vector_base.wv.init_sims(replace=True)
    # for i in sorted(words_ids_trim_list, reverse=True):
    #     del(vector_base.wv.index2word[i])
    # vector_base.save('data/bio_word2vec_trim')
    restrict_w2v(vector_base,words_trim_list)
    return vocab_count_dict, words_trim_list, gaz_position_dict


def get_text_from_raw_file(datas_file):
    text_list = []
    with open(datas_file, mode='r', encoding='utf-8') as fr:
        for index, line in enumerate(fr):
            line = line.strip("\n")
            if index % 2 == 0:
                char_list = line.split(" ")
                merge_text = "".join(char_list)
                text_list.append(merge_text)
    return text_list


def restrict_w2v(vector_base, restricted_word_set):
    """
    根据词表，将频率较高的词取出来，减小embedding的规模
    :param vector_base:
    :param restricted_word_set:
    :return:
    """
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    # new_vectors_norm = []

    for i in range(len(vector_base.vocab)):
        word = vector_base.index2entity[i]
        vec = vector_base.vectors[i]
        vocab = vector_base.vocab[word]
        # vec_norm = vector_base.vectors_norm[i]
        if word not in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            # new_vectors_norm.append(vec_norm)

    vector_base.vocab = new_vocab
    vector_base.vectors = np.array(new_vectors)
    vector_base.index2entity = new_index2entity
    vector_base.index2word = new_index2entity
    # vector_base.vectors_norm = new_vectors_norm
    vector_base.init_sims(replace=True)
    # vector_base.save('bio_word2vec')
    vector_base.save('data/bio_word2vec_trim')


if __name__ == "__main__":
    # word2vec = KeyedVectors.load_word2vec_format("data/sgns.financial.bigram-char", binary=False,unicode_errors='ignore')
    # word2vec.init_sims(replace=True)

    embedding_file = "data/sgns.financial.bigram-char"
    training_context_datas = get_text_from_raw_file("data/orig_data_train.txt")
    training_context_datas.extend(get_text_from_raw_file("data/orig_data_dev.txt"))
    training_context_datas.extend(get_text_from_raw_file("data/orig_data_test.txt"))
    vocab_count_dict, words_trim_list, gaz_position_dict = get_word_tf(embedding_file, training_context_datas)
    # vocab_count_dict = {key: value for key, value in vocab_count_dict.items() if value > 0}

    # with open("data/soft_vocab_count.json",mode='r',encoding='utf-8') as fr:
    #     vocab_count_dict = json.load(fr)
    vocab_count_dict = {key: value for key, value in vocab_count_dict.items() if value >= 5}
    with open("data/soft_vocab_count.json", mode='w', encoding='utf-8') as fw1:
        json.dump(vocab_count_dict, fw1, ensure_ascii=False, indent=4)
    with open("data/gaz_position_dict.json", mode='w', encoding='utf-8') as fw2:
        json.dump(gaz_position_dict, fw2, ensure_ascii=False, indent=4)

