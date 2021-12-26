import os
import json
import datetime
import copy
import codecs
import numpy as np
import gensim
import tensorflow as tf
from data_processing.basic_prepare_data import BaseDataPreparing
from data_processing.mrc_query_map import ner_query_map


class LEARPrepareData(BaseDataPreparing):
    def __init__(self, vocab_file, slot_file, config, bert_file, max_length, gen_new_data=False, is_inference=False,
                 add_soft_lexicon=False, vocab_count_file=None, embedding_file=None, gaze_vocab_file=None):
        self.bert_file = bert_file
        self.max_length = max_length
        self.add_soft_lexicon = add_soft_lexicon
        if add_soft_lexicon:
            self.vocab_count_dict = self.load_vocab_count(vocab_count_file)
            with open(gaze_vocab_file, mode='r', encoding='utf-8') as fr:
                self.gaze_vocab_dict = json.load(fr)
            self.word2vec = gensim.models.KeyedVectors.load(embedding_file, mmap='r')
        self.embedding_dim = 300
        self.scale = np.sqrt(3.0 / self.embedding_dim)
        # self.label_tokens_ids_list,label_token_type_ids_list,label_token_masks_list,label_tokens_length_list = self.gen_label_tokens()
        super(LEARPrepareData, self).__init__(vocab_file, slot_file, config, pretrained_embedding_file=None,
                                              word_embedding_file=None, load_w2v_embedding=False,
                                              load_word_embedding=False, gen_new_data=gen_new_data,
                                              is_inference=is_inference)

    def load_vocab_count(self, vocab_count_file):
        with open(vocab_count_file, mode='r', encoding='utf-8') as fr:
            vocab_count_dict = json.load(fr)
        return vocab_count_dict

    def init_final_data_path(self, config, load_word_embedding=False):
        # self.train_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_text_name"))
        # self.valid_X_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_text_name"))
        # self.train_start_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("train_data_start_tag_name"))
        # self.train_end_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"),
        #                                        config.get("train_data_end_tag_name"))
        # self.valid_start_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_start_tag_name"))
        # self.valid_end_Y_path = os.path.join(config.get("data_dir"),config.get("train_valid_data_dir"), config.get("valid_data_end_tag_name"))
        # self.test_X_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_text_name"))
        # self.train_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_token_type_ids_name"))
        # self.valid_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_token_type_ids_name"))
        # self.test_token_type_ids_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_token_type_ids_name"))

        # self.train_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("train_data_query_len_name"))
        # self.valid_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("valid_data_query_len_name"))
        # self.test_query_len_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_query_len_name"))
        # self.test_query_class_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_query_class"))
        # self.src_test_sample_id_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_src_sample_id"))
        pass

        # self.test_Y_path = os.path.join(config.get("data_dir"), config.get("train_valid_data_dir"), config.get("test_data_tag_name"))

    def split_one_sentence_based_on_length(self, texts, text_allow_length, labels=None):
        # 通用的截断
        data_list = []
        data_label_list = []
        if len(texts) > text_allow_length:
            left_length = 0
            while left_length + text_allow_length < len(texts):
                cur_cut_index = left_length + text_allow_length
                if labels != None:
                    last_label_tmp = labels[cur_cut_index - 1]
                    if last_label_tmp.upper() != "O":
                        while labels[cur_cut_index - 1].upper() != "O":
                            cur_cut_index -= 1
                    data_label_list.append(labels[left_length:cur_cut_index])
                data_list.append(texts[left_length:cur_cut_index])
                left_length = cur_cut_index

            # 别忘了最后还有余下的一小段没处理
            if labels != None:
                data_label_list.append(labels[left_length:])
            data_list.append(texts[left_length:])
        else:
            data_list.append(texts)
            data_label_list.append(labels)
        #         if len(data_list) > 2:
        #             for data in data_list:
        #                 print("".join(data))
        return data_list, data_label_list

    def split_one_sentence_based_on_entity_direct(self, texts, text_allow_length, labels=None):
        # 对于超过长度的直接截断
        data_list = []
        data_label_list = []
        if len(texts) > text_allow_length:
            pick_texts = texts[0:text_allow_length]
            data_list.append(pick_texts)
            if labels != None:
                data_label_list.append(labels[0:text_allow_length])
                return data_list, data_label_list
        else:
            data_list.append(texts)
            data_label_list.append(labels)
            return data_list, data_label_list

    # def tranform_singlg_data_example(self,text,slot_label):
    #     # print(text)
    #     word_list = []
    #     # if not self.label_less:
    #     #     word_list.append("[CLS]")
    #     word_list.append("[CLS]")
    #
    #     if self.is_inference:
    #         word_list.extend([w for w in text if w !=" "])
    #     else:
    #         word_list.extend(self.tokenizer.tokenize(text))
    #     if len(word_list)>=self.max_length:
    #         word_list = word_list[0:self.max_length-1]
    #
    #     # if not self.label_less:
    #     #     word_list.append("[SEP]")
    #     word_list.append("[SEP]")
    #     word_id_list = self.tokenizer.convert_tokens_to_ids(word_list)
    #     # print(len(word_id_list))
    #     return word_id_list

    def token_data_text(self, text):
        word_list = []
        if self.is_inference:
            word_list.extend([w for w in text if w != " "])
        else:
            word_list.extend(self.tokenizer.tokenize(text))
        return word_list

    def gen_lexicon_embedding_for_text(self, token_ids):
        orig_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        soft_lexicon_embeddings_list = []
        start = datetime.datetime.now()
        for index, orig_token in enumerate(orig_tokens):
            if len(orig_token) > 1:
                # word_piece切词可能的情况 ##
                if orig_token.startswith("##"):
                    orig_token = orig_token[2:]
                # if orig_token in ["[CLS]","[SEP]","[PAD]","[UNK]"]:

            token_word_embeddings = {"B": [], "M": [], "E": [], "S": []}
            if orig_token in self.gaze_vocab_dict:
                cur_gaze_infos = self.gaze_vocab_dict.get(orig_token)
                if "B" in cur_gaze_infos:
                    for j in range(index + 1, len(orig_tokens)):
                        if j == len(orig_tokens) - 1:
                            cur_span = orig_tokens[index:]
                        else:
                            cur_span = orig_tokens[index:j + 1]
                        if cur_span in cur_gaze_infos["B"]:
                            token_word_embeddings["B"].append((self.vocab_count_dict[cur_span],
                                                               self.word2vec[cur_span] * self.vocab_count_dict[
                                                                   cur_span]))
                            break
                if "E" in cur_gaze_infos:
                    for j in range(0, index):
                        if index == len(orig_tokens) - 1:
                            cur_span = orig_tokens[j:]
                        else:
                            cur_span = orig_tokens[j:index + 1]
                        if cur_span in cur_gaze_infos["E"]:
                            token_word_embeddings["E"].append((self.vocab_count_dict[cur_span],
                                                               self.word2vec[cur_span] * self.vocab_count_dict[
                                                                   cur_span]))
                            break
                if "S" in cur_gaze_infos:
                    cur_span = orig_token
                    token_word_embeddings["S"].append(
                        (self.vocab_count_dict[cur_span], self.word2vec[cur_span] * self.vocab_count_dict[cur_span]))

                if "M" in cur_gaze_infos:
                    if index > 0 and index < len(orig_tokens) - 1:
                        candi_word_list = cur_gaze_infos["M"]
                        for candi_word in candi_word_list:
                            word_len = len(candi_word)
                            char_list = [w for w in candi_word]
                            char_index = char_list.index(orig_token)
                            if char_index > index or index - char_index + word_len > len(orig_tokens):
                                continue
                            pick_span = orig_tokens[index - char_index:index - char_index + word_len]
                            if candi_word == pick_span:
                                token_word_embeddings["M"].append((self.vocab_count_dict[candi_word],
                                                                   self.word2vec[candi_word] * self.vocab_count_dict[
                                                                       candi_word]))

            # for key in token_word_embeddings:
            #     if key == "B":
            #         # 出现在词首
            #         for vocab_candi in self.vocab_count_dict:
            #             if vocab_candi.startswith(orig_token) and vocab_candi != orig_token:
            #                 token_word_embeddings["B"].append((self.vocab_count_dict[vocab_candi],np.array(self.word2vec[vocab_candi])*self.vocab_count_dict[vocab_candi]))
            #     elif key == "M":
            #         for vocab_candi in self.vocab_count_dict:
            #             if not vocab_candi.startswith(orig_token) and not vocab_candi.endswith(orig_token) and vocab_candi.__contains__(orig_token):
            #                 token_word_embeddings["M"].append((self.vocab_count_dict[vocab_candi],np.array(self.word2vec[vocab_candi])*self.vocab_count_dict[vocab_candi]))
            #     elif key == "E":
            #         for vocab_candi in self.vocab_count_dict:
            #             if vocab_candi.endswith(orig_token) and vocab_candi != orig_token:
            #                 token_word_embeddings["E"].append((self.vocab_count_dict[vocab_candi],np.array(self.word2vec[vocab_candi])*self.vocab_count_dict[vocab_candi]))
            #     else:
            #         for vocab_candi in self.vocab_count_dict:
            #             if vocab_candi == orig_token:
            #                 token_word_embeddings["S"].append((self.vocab_count_dict[vocab_candi],np.array(self.word2vec[vocab_candi])*self.vocab_count_dict[vocab_candi]))
            # # end = datetime.datetime.now()
            # # print("qqq")
            # # print((end-start).microseconds)
            final_embeddings_for_cur_token = []
            # start = datetime.datetime.now()
            for key in token_word_embeddings:
                if token_word_embeddings[key]:
                    cur_embedding_result = np.sum([ele[1] for ele in token_word_embeddings[key]], axis=0, keepdims=True)
                    total_count = np.sum([ele[0] for ele in token_word_embeddings[key]])
                    cur_embedding_result = 4 * cur_embedding_result / total_count
                else:
                    cur_embedding_result = np.random.uniform(-self.scale, self.scale, [1, self.embedding_dim])
                final_embeddings_for_cur_token.append(cur_embedding_result)
            # end = datetime.datetime.now()
            # print("adf")
            # print((end-start).microseconds)
            final_embedding_cur_token = np.concatenate(final_embeddings_for_cur_token, axis=-1)
            soft_lexicon_embeddings_list.append(final_embedding_cur_token)
        soft_lexicon_embeddings_list = np.concatenate(soft_lexicon_embeddings_list, axis=0)
        end = datetime.datetime.now()
        # print("end")
        # print((end-start).seconds)
        return soft_lexicon_embeddings_list

    def gen_label_tokens(self, label_max_len=24):
        # labels = en_conll03_ner.get("tags")
        # label_2_ids = {ele:idx for idx,ele in enumerate(labels)}
        # id2_2_label = {idx:ele for idx,ele in enumerate(labels)}
        label_tokens_ids_list = []
        label_token_type_ids_list = []
        label_token_masks_list = []
        label_tokens_length_list = []
        label_soft_embeddings_list = []
        for slot_tag in ner_query_map.get("tags"):
            slot_query = ner_query_map.get("natural_query").get(slot_tag)
            slot_query_tokenize = [w for w in slot_query]
            slot_query_tokenize.insert(0, "[CLS]")
            slot_query_tokenize.append("[SEP]")
            cur_label_tokens_ids = self.tokenizer.convert_tokens_to_ids(slot_query_tokenize)
            label_tokens_ids_list.append(cur_label_tokens_ids)
            label_token_type_ids_list.append([0] * len(cur_label_tokens_ids))
            label_tokens_length_list.append(len(cur_label_tokens_ids))
            label_token_masks_list.append([1] * len(cur_label_tokens_ids))
            if self.add_soft_lexicon:
                soft_lexicon_embeddings = self.gen_lexicon_embedding_for_text(cur_label_tokens_ids)
                label_soft_embeddings_list.append(soft_lexicon_embeddings)
            else:
                soft_lexicon_embeddings = np.random.uniform(size=[label_max_len, 1200])
                label_soft_embeddings_list.append(soft_lexicon_embeddings)

        # max_len = max(label_tokens_length_list)
        # max_len = 24
        for i in range(len(label_tokens_ids_list)):
            while len(label_tokens_ids_list[i]) < label_max_len:
                label_tokens_ids_list[i].append(0)
            while len(label_token_type_ids_list[i]) < label_max_len:
                label_token_type_ids_list[i].append(0)
            while len(label_token_masks_list[i]) < label_max_len:
                label_token_masks_list[i].append(0)
            # while len(label_soft_embeddings_list[i]) < label_max_len:

            if self.add_soft_lexicon:
                input_len = len(label_soft_embeddings_list[i])
                label_soft_embeddings_list[i] = np.pad(label_soft_embeddings_list[i],
                                                       ((0, label_max_len - input_len), (0, 0)))

        return label_tokens_ids_list, label_token_type_ids_list, label_token_masks_list, label_tokens_length_list, label_soft_embeddings_list

    def find_tag_start_end_index(self, tag_list, label_list):
        # 输出[[1,0,0],[0,1,0],[0,0,1]]
        start_label_list = []
        end_label_list = []
        # start_tag = "B-"+tag
        # end_tag = "I-"+tag
        for i in range(len(label_list)):
            cur_start_tags = [0] * len(tag_list)
            cur_end_tags = [0] * len(tag_list)
            for index, tag in enumerate(tag_list):
                start_tag = "B-" + tag
                end_tag = "I-" + tag
                if label_list[i].upper() == start_tag:
                    # begin
                    cur_start_tags[index] = 1
                elif label_list[i].upper() == end_tag:
                    if i == len(label_list) - 1:
                        # last tag
                        cur_end_tags[index] = 1
                    else:
                        if label_list[i + 1].upper() != end_tag:
                            cur_end_tags[index] = 1
            assert sum(cur_start_tags) < 2
            assert sum(cur_end_tags) < 2
            start_label_list.append(cur_start_tags)
            end_label_list.append(cur_end_tags)
        return start_label_list, end_label_list
        # start_index_tag = [0] * len(label_list)
        # end_index_tag = [0] * len(label_list)
        # start_tag = "B-"+tag
        # end_tag = "I-"+tag
        # for i in range(len(start_index_tag)):
        #     if label_list[i].upper() == start_tag:
        #         # begin
        #         start_index_tag[i] = 1
        #     elif label_list[i].upper() == end_tag:
        #         if i == len(start_index_tag)-1:
        #             # last tag
        #             end_index_tag[i] = 1
        #         else:
        #             if label_list[i+1].upper() != end_tag:
        #                 end_index_tag[i] = 1
        # return start_index_tag,end_index_tag

    def trans_orig_data_to_training_data(self, datas_file):
        data_X = []
        data_start_Y = []
        data_end_Y = []
        token_type_ids_list = []
        input_masks_list = []
        lexicon_embeddings_list = []
        labels_set = ner_query_map.get("tags")
        with codecs.open(datas_file, 'r', 'utf-8') as fr:
            tmp_text_split = None
            for index, line in enumerate(fr):
                if index % 10 == 0:
                    print(index)
                line = line.strip("\n")
                if index % 2 == 0:
                    tmp_text_split = self.token_data_text(line)
                else:
                    slot_label_list = self.tokenizer.tokenize(line)
                    gen_tmp_X_texts, gen_tmp_y_labels = self.split_one_sentence_based_on_entity_direct(
                        tmp_text_split, self.max_length - 2, slot_label_list)
                    for tmp_X, tmp_Y in zip(gen_tmp_X_texts, gen_tmp_y_labels):
                        tmp_X.insert(0, "[CLS]")
                        tmp_X.append("[SEP]")
                        input_tokens_ids = self.tokenizer.convert_tokens_to_ids(tmp_X)
                        if self.add_soft_lexicon:
                            soft_lexicon_embeddings = self.gen_lexicon_embedding_for_text(input_tokens_ids)
                        else:
                            soft_lexicon_embeddings = np.random.uniform(size=[len(input_tokens_ids), 1200])
                        lexicon_embeddings_list.append(soft_lexicon_embeddings)
                        token_type_ids = [0] * (len(input_tokens_ids))
                        input_masks = [1] * len(input_tokens_ids)

                        start_labels, end_labels = self.find_tag_start_end_index(labels_set, tmp_Y)
                        start_labels.insert(0, [0, 0, 0])
                        start_labels.append([0, 0, 0])
                        end_labels.insert(0, [0, 0, 0])
                        end_labels.append([0, 0, 0])
                        data_X.append(input_tokens_ids)
                        token_type_ids_list.append(token_type_ids)
                        input_masks_list.append(input_masks)
                        data_start_Y.append(start_labels)
                        data_end_Y.append(end_labels)

        return data_X, data_start_Y, data_end_Y, token_type_ids_list, input_masks_list, lexicon_embeddings_list

    def gen_train_data_from_raw(self, type="train"):
        if type == "train":
            data_X, data_start_Y, data_end_Y, token_type_ids_list, input_masks_list, soft_embeddings_list = self.trans_orig_data_to_training_data(
                self.train_data_file)
        else:
            data_X, data_start_Y, data_end_Y, token_type_ids_list, input_masks_list, soft_embeddings_list = self.trans_orig_data_to_training_data(
                self.dev_data_file)
        return data_X, data_start_Y, data_end_Y, token_type_ids_list, input_masks_list, soft_embeddings_list

    def gen_data_for_prediction(self, datas_file):
        text_allow_max_len = self.max_length - 2
        all_token_ids_list = []
        all_token_type_ids_list = []
        all_input_masks_list = []
        all_input_lens_list = []
        all_lexicon_embeddings_list = []
        with codecs.open(datas_file, 'r', 'utf-8') as fr:
            tmp_text_split = None
            for index, line in enumerate(fr):
                line = line.strip("\n")
                if index % 2 == 0:
                    tmp_text_split = self.token_data_text(line)
                else:
                    # slot_label_list = self.tokenizer.tokenize(line)
                    cur_sample_token_ids = []
                    cur_sample_token_type_ids = []
                    cur_sample_input_masks = []
                    cur_sample_input_length = []
                    cur_sample_soft_embeddings = []
                    gen_tmp_X_texts, _ = self.split_one_sentence_based_on_length(
                        tmp_text_split, text_allow_max_len)
                    for tmp_X in gen_tmp_X_texts:
                        tmp_X.insert(0, "[CLS]")
                        tmp_X.append("[SEP]")
                        input_tokens_ids = self.tokenizer.convert_tokens_to_ids(tmp_X)
                        token_type_ids = [0] * (len(input_tokens_ids))
                        input_masks = [1] * len(input_tokens_ids)
                        cur_sample_token_ids.append(input_tokens_ids)
                        cur_sample_token_type_ids.append(token_type_ids)
                        cur_sample_input_masks.append(input_masks)
                        cur_sample_input_length.append(len(input_tokens_ids))
                        if self.add_soft_lexicon:
                            soft_lexicon_embeddings = self.gen_lexicon_embedding_for_text(input_tokens_ids)
                            cur_sample_soft_embeddings.append(soft_lexicon_embeddings)
                    all_token_ids_list.append(cur_sample_token_ids)
                    all_token_type_ids_list.append(cur_sample_token_type_ids)
                    all_input_masks_list.append(cur_sample_input_masks)
                    all_input_lens_list.append(cur_sample_input_length)
                    all_lexicon_embeddings_list.append(cur_sample_soft_embeddings)
        return all_token_ids_list, all_token_type_ids_list, all_input_masks_list, all_input_lens_list, all_lexicon_embeddings_list

    def gen_train_dev_from_orig_data(self, gen_new):
        # if gen_new:
        #     train_data_X,train_data_start_Y,train_data_end_Y,train_token_type_ids_list,train_query_len_list = self.trans_orig_data_to_training_data(self.train_data_file)
        #     dev_data_X,dev_data_start_Y,dev_data_end_Y,dev_token_type_ids_list,dev_query_len_list = self.trans_orig_data_to_training_data(self.dev_data_file)
        #     # test_data_X,test_data_start_Y,test_data_end_Y,test_token_type_ids_list,test_query_len_list = self.trans_orig_data_to_training_data(self.test_data_file)
        #     # dev_data_X = np.concatenate((dev_data_X,test_data_X),axis=0)
        #     # dev_data_Y = np.concatenate((dev_data_Y,test_data_Y),axis=0)
        #     self.train_samples_nums = len(train_data_X)
        #     self.eval_samples_nums = len(dev_data_X)
        #     np.save(self.train_X_path, train_data_X)
        #     np.save(self.valid_X_path, dev_data_X)
        #     np.save(self.train_start_Y_path,train_data_start_Y)
        #     np.save(self.train_end_Y_path, train_data_end_Y)
        #     np.save(self.valid_start_Y_path, dev_data_start_Y)
        #     np.save(self.train_start_Y_path, train_data_start_Y)
        #     np.save(self.valid_end_Y_path,dev_data_end_Y)
        #     np.save(self.train_token_type_ids_path, train_token_type_ids_list)
        #     np.save(self.valid_token_type_ids_path, dev_token_type_ids_list)
        #     np.save(self.train_query_len_path,train_query_len_list)
        #     np.save(self.valid_query_len_path,dev_query_len_list)
        #     # np.save(self.test_X_path,test_data_X)
        #     # np.save(self.test_token_type_ids_path, test_token_type_ids_list)
        #     # np.save(self.test_query_len_path, test_query_len_list)
        #     # np.save(self.test_Y_path,test_data_Y)
        # else:
        #     train_data_X = np.load(self.train_X_path)
        #     dev_data_X = np.load(self.valid_X_path)
        #     self.train_samples_nums = len(train_data_X)
        #     self.eval_samples_nums = len(dev_data_X)
        pass

    # def trans_test_data(self):
    #         self.gen_test_data_from_orig_data(self.test_data_file)

    # def gen_test_data_from_orig_data(self,datas_file):
    #     # 相对于训练集来说，测试集构造数据要更复杂一点
    #     # 1、query要标明，2、因长度问题分割的句子最后要拼起来，因此同一个原样本的要标明 3、最后要根据query对应的实体类别根据start end 关系拼起来
    #     data_X = []
    #     token_type_ids_list = []
    #     query_len_list = []
    #     query_class_list = []
    #     src_test_sample_id = []
    #     with codecs.open(datas_file, 'r', 'utf-8') as fr:
    #         tmp_text_split = None
    #         for index, line in enumerate(fr):
    #             line = line.strip("\n")
    #             if index % 2 == 0:
    #                 tmp_text_split = self.token_data_text(line)
    #                 cur_sample_id = int(index / 2)
    #                 for slot_tag in self.query_map_dict:
    #                     slot_query = self.query_map_dict.get(slot_tag)
    #                     slot_query = [w for w in slot_query]
    #                     query_len = len(slot_query)
    #                     text_allow_max_len = self.max_length - query_len
    #                     gen_tmp_X_texts, _ = self.split_one_sentence_based_on_length(
    #                         tmp_text_split, text_allow_max_len)
    #                     for tmp_X in gen_tmp_X_texts:
    #                         x_merge = slot_query + tmp_X
    #                         token_type_ids = [0] * len(slot_query) + [1] * (len(tmp_X))
    #                         x_merge = self.tokenizer.convert_tokens_to_ids(x_merge)
    #                         data_X.append(x_merge)
    #                         src_test_sample_id.append(cur_sample_id)
    #                         query_class_list.append(en_conll03_ner.get("tags").index(slot_tag))
    #                         token_type_ids_list.append(token_type_ids)
    #                         query_len_list.append(query_len)
    #     np.save(self.test_X_path,data_X)
    #     np.save(self.test_token_type_ids_path, token_type_ids_list)
    #     np.save(self.test_query_len_path, query_len_list)
    #     np.save(self.test_query_class_path,query_class_list)
    #     np.save(self.src_test_sample_id_path,src_test_sample_id)


def data_generator_lear(input_Xs, input_label_tokens, input_token_type_ids, label_token_type_ids, input_masks,
                        label_masks, label_token_lens, start_Ys, end_Ys, input_lexicons_list, label_lexicons):
    for index in range(len(input_Xs)):
        input_x = input_Xs[index]
        # label_token = input_label_tokens[index]
        start_y = start_Ys[index]
        end_y = end_Ys[index]
        input_token_type_id = input_token_type_ids[index]
        # label_token_type_id = label_token_type_ids[index]
        input_mask = input_masks[index]
        # label_mask = label_masks[index]
        input_len = len(input_x)
        input_lexicons = input_lexicons_list[index]
        # print(label_token)
        # print(label_token_type_id)
        # print(label_mask)
        yield (input_x, input_label_tokens, label_token_type_ids, input_token_type_id, input_len, label_token_lens,
               input_mask, label_masks, input_lexicons, label_lexicons), (start_y, end_y)


def input_lear_fn(input_Xs, input_label_tokens, input_token_type_ids, label_token_type_ids, input_masks, label_masks,
                  start_Ys, end_Ys, label_token_lens, input_lexicons_list, label_lexicons, is_training, is_testing,
                  args):
    ##input_ids, label_token_ids, label_token_type_ids, token_type_ids_list, text_length_list, label_token_length, token_masks, label_token_masks
    _shapes = (
    ([None], [3, 24], [3, 24], [None], (), (3), [None], [3, 24], [None, 1200], [3, 24, 1200]), ([None, 3], [None, 3]))
    _types = ((tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32),
              (tf.int32, tf.int32))
    _pads = ((0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0), (0, 0))
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator_lear(input_Xs, input_label_tokens, input_token_type_ids, label_token_type_ids,
                                    input_masks, label_masks, label_token_lens, start_Ys, end_Ys, input_lexicons_list,
                                    label_lexicons),
        output_shapes=_shapes,
        output_types=_types, )
    if is_training:
        ds = ds.shuffle(args.shuffle_buffer).repeat()
    if is_training:
        ds = ds.padded_batch(args.train_batch_size, _shapes, _pads)
    else:
        if is_testing:
            ds = ds.padded_batch(args.test_batch_size, _shapes, _pads)
        else:
            ds = ds.padded_batch(args.valid_batch_size, _shapes, _pads)
    ds = ds.prefetch(args.pre_buffer_size)

    return ds
