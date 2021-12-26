import tensorflow as tf
import datetime
import re
import os
import numpy as np
from configs.lear_config import lear_config
from data_processing.mrc_query_map import ner_query_map
from data_processing.lear_processor import LEARPrepareData
from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from sklearn_crfsuite import metrics
from tensorflow.contrib import predictor


def gen_orig_test_text_label(orig_test_file, has_cls=False):
    orig_text = []
    orig_label = []
    with open(orig_test_file, mode='r', encoding='utf-8') as fr:
        for i, line in enumerate(fr):
            if i % 2 == 0:
                text = line.strip("\n")
                text_split = text.split(" ")
                if has_cls:
                    print(text_split)
                    print(has_cls)
                    text_split.insert(0, "[CLS]")
                    text_split.append("[SEP]")
                orig_text.append(text_split)
            else:
                text = line.strip("\n")
                text_split = text.split(" ")
                if has_cls:
                    text_split.insert(0, "O")
                    text_split.append("O")
                orig_label.append(text_split)
    return orig_text, orig_label


def gen_entity_from_label_id_list(text_lists, label_id_list, id2slot_dict, orig_test=False):
    """
    B-LOC
    B-PER
    B-ORG
    I-LOC
    I-ORG
    I-PER
    :param label_id_list:
    :param id2slot_dict:
    :return:
    """
    entity_list = []
    for outer_idx, label_ids in enumerate(label_id_list):
        each_sample_entity_list = []
        start_index_list = [index for index, label_id in enumerate(label_ids) if label_id.startswith("B")]
        for start_index in start_index_list:
            if start_index == len(label_ids) - 1:
                cur_type = label_ids[start_index].split("-")[-1]
                each_sample_entity_list.append(cur_type + text_lists[outer_idx][start_index])
            for idx in range(start_index + 1, len(label_ids)):
                if label_ids[idx].startswith("B") or label_ids[idx] == "O":
                    if start_index + 1 == idx:
                        cur_type = label_ids[start_index].split("-")[-1]
                        each_sample_entity_list.append(cur_type + text_lists[outer_idx][start_index])
                    else:
                        cur_type = label_ids[start_index].split("-")[-1]
                        each_sample_entity_list.append(cur_type + "".join(text_lists[outer_idx][start_index:idx]))
                    break
        each_sample_entity_list = list(set(each_sample_entity_list))
        entity_list.append(each_sample_entity_list)
    return entity_list


def cal_mertric_from_two_list(prediction_list, true_list):
    tp, fp, fn = 0, 0, 0
    for pred_entity, true_entity in zip(prediction_list, true_list):
        pred_entity_set = set(pred_entity)
        true_entity_set = set(true_entity)
        tp += len(true_entity_set & pred_entity_set)
        fp += len(pred_entity_set - true_entity_set)
        fn += len(true_entity_set - pred_entity_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec)
    print("span_level pre micro_avg:{}".format(prec))
    print("span_level rec micro_avg:{}".format(rec))
    print("span_level f1 micro_avg:{}".format(f1))


class fastPredict(object):
    def __init__(self, model_path, add_soft_embeddings, config):
        self.model_path = model_path
        self.add_soft_embeddings = add_soft_embeddings
        self.data_loader = self.init_data_loader(config)
        self.predict_fn = self.load_models()
        self.config = config
        self.label_list = ner_query_map.get("tags")

    def init_data_loader(self, config):
        vocab_file_path = os.path.join(config.get("bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(config.get("slot_list_root_path"), config.get("bert_slot_complete_file_name"))
        bert_config_file = os.path.join(config.get("bert_pretrained_model_path"), config.get("bert_config_path"))

        data_loader = LEARPrepareData(vocab_file_path, slot_file, lear_config, bert_config_file, 384, True, False,
                                      self.add_soft_embeddings, "data/soft_vocab_count.json", "data/bio_word2vec_trim",
                                      "data/gaz_position_dict.json")
        return data_loader

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def predict_lear(self, text_token_ids=None, token_type_ids=None, input_masks=None, token_lens=None,
                     input_soft_embeddings=None, label_token_ids=None, label_token_type_ids=None,
                     label_input_masks=None, label_token_lens=None, label_soft_embeddings=None, args=None):
        # features['words'],features["label_tokens"],features["label_token_type_ids"],features['token_type_ids'],features['text_length'],features['label_text_length'],features['token_masks'],features['label_token_masks']
        if self.add_soft_embeddings:
            predictions = self.predict_fn({'words': [text_token_ids], 'label_tokens': label_token_ids,
                                           'label_token_type_ids': label_token_type_ids,
                                           'token_type_ids': [token_type_ids], 'text_length': [token_lens],
                                           'label_text_length': label_token_lens,
                                           'token_masks': [input_masks], 'label_token_masks': label_input_masks,
                                           'input_lexicons': [input_soft_embeddings],
                                           'label_lexicons': label_soft_embeddings})
        else:
            predictions = self.predict_fn({'words': [text_token_ids], 'label_tokens': label_token_ids,
                                           'label_token_type_ids': label_token_type_ids,
                                           'token_type_ids': [token_type_ids], 'text_length': [token_lens],
                                           'label_text_length': label_token_lens,
                                           'token_masks': [input_masks], 'label_token_masks': label_input_masks})
        # test_input_fn = lambda: input_bert_mrc_fn(text, start_id_fake,start_id_fake,token_type_ids,query_len,False,True,args)
        # features['words'],features['text_length'],features['query_length'],features['token_type_ids']
        # predictions = self.predict_fn.predict(test_input_fn)
        # print(predictions)
        start_probs, end_probs = predictions.get("start_probs")[0], predictions.get("end_probs")[0]
        # print(start_ids)
        return start_probs, end_probs

    def extract_entity_from_start_end_ids(self, text_tokens, start_ids, end_ids):
        # start_ids 预测为1的起始位置index list
        # end_ids 预测为1的终止位置index list
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        # text_cur_index = 0
        for i, start_index in enumerate(start_ids):
            cur_ends = end_ids[end_ids >= start_index]
            if len(cur_ends) > 0:
                end_idx = cur_ends[0]
                if i < len(start_ids) - 1:
                    # 当前end_id 不能超过下一个start_id
                    if end_idx < start_ids[i + 1]:
                        span_cur = text_tokens[start_index:end_idx + 1]
                        span_cur = "".join(span_cur)
                    else:
                        span_cur = text_tokens[start_index]

                else:
                    span_cur = text_tokens[start_index:end_idx + 1]
                    span_cur = "".join(span_cur)
                entity_list.append(span_cur)
            else:
                span_cur = text_tokens[start_index]
                entity_list.append(span_cur)

        entity_list = list(set(entity_list))
        return entity_list

    def extract_entity_from_start_end_ids_her(self, text_tokens, start_ids, end_ids, cur_cate_start_probs,
                                              cur_cate_end_probs):
        # start_ids 预测为1的起始位置index list
        # end_ids 预测为1的终止位置index list
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        # text_cur_index = 0
        # state = 0 # 还未定位到start和end
        # # state_2 = False # 定位到start
        # # state_3 = False # start和end都定位到了
        # a_s = -1
        # a_e = -1
        if len(end_ids) == 0:
            # 单字成实体
            for idx in start_ids:
                entity_list.append(text_tokens[idx])
        else:
            # for i,start_index in enumerate(start_ids):
            #     cur_ends = end_ids[end_ids >= start_index]
            #     if len(cur_ends) > 0:
            #         end_idx = cur_ends[0]
            #         if i < len(start_ids)-1:
            # pass
            start_prob, end_prob, start_idx, end_idx = -1, -1, -1, -1
            for i in range(len(text_tokens)):
                if end_idx != -1:
                    if end_idx == len(text_tokens) - 1:
                        entity_list.append("".join(text_tokens[start_idx:]))
                    else:
                        entity_list.append("".join(text_tokens[start_idx:end_idx + 1]))
                    start_prob, end_prob, start_idx, end_idx = -1, -1, -1, -1
                if cur_cate_start_probs[i] > start_prob:
                    start_prob = cur_cate_start_probs[i]
                    start_idx = i
                if i in end_ids and start_idx in start_ids:
                    end_prob = cur_cate_end_probs[i]
                    end_idx = i
            if end_idx != -1:
                if end_idx == len(text_tokens) - 1:
                    entity_list.append("".join(text_tokens[start_idx:]))
                else:
                    entity_list.append("".join(text_tokens[start_idx:end_idx + 1]))
        # for i,start_index in enumerate(start_ids):
        #     cur_ends = end_ids[end_ids >= start_index]
        #     if len(cur_ends) > 0:
        #         end_idx = cur_ends[0]
        #         if i < len(start_ids)-1:
        #             # 当前end_id 不能超过下一个start_id,根据start_prob较大的来选择
        #             if end_idx < start_ids[i+1]:

        #                 span_cur = text_tokens[start_index:end_idx+1]
        #                 span_cur = "".join(span_cur)
        #             else:
        #                 span_cur = text_tokens[start_index]

        #         else:
        #             span_cur = text_tokens[start_index:end_idx+1]
        #             span_cur = "".join(span_cur)
        #         entity_list.append(span_cur)
        #     else:
        #         span_cur = text_tokens[start_index]
        #         entity_list.append(span_cur)

        entity_list = list(set(entity_list))
        return entity_list

    def extract_span_from_start_end(self, start_probs, end_probs, token_ids, text_len, s_limit=0.5):
        label_num = start_probs.shape[-1]
        tokens = self.data_loader.tokenizer.convert_ids_to_tokens(token_ids)
        entity_type_result_list = []
        for label_idx in range(label_num):
            # 3个类别
            cur_start_idxes = np.where(
                start_probs[:text_len, label_idx] > s_limit)
            cur_end_idxes = np.where(end_probs[:text_len, label_idx] > s_limit)
            if cur_start_idxes[0].size == 0:
                continue
            cur_start_idxes = cur_start_idxes[0]
            cur_end_idxes = cur_end_idxes[0]
            # cur_cate_start_probs = start_probs[:text_len,label_idx]
            # cur_cate_end_probs = end_probs[:text_len,label_idx]
            entity_list = self.extract_entity_from_start_end_ids(tokens, cur_start_idxes, cur_end_idxes)
            # entity_list = self.extract_entity_from_start_end_ids_her(tokens,cur_start_idxes,cur_end_idxes,cur_cate_start_probs,cur_cate_end_probs)
            if len(entity_list) > 0:
                entity_with_type_list = [self.label_list[label_idx] + entity_str for entity_str in entity_list]
                entity_type_result_list.extend(entity_with_type_list)
        return entity_type_result_list

    def predict_for_all_data(self, test_data_file):
        test_all_token_ids_list, test_all_token_type_ids_list, test_all_input_masks_list, test_all_input_lens_list, test_all_soft_embeddings = self.data_loader.gen_data_for_prediction(
            test_data_file)
        label_tokens_ids_list, label_token_type_ids_list, label_token_masks_list, label_tokens_length_list, label_soft_embeddings = self.data_loader.gen_label_tokens()
        prediction_result_list = []
        begin = datetime.datetime.now()
        for i in range(len(test_all_token_ids_list)):
            # 遍历每个样本
            cur_sample_result_list = []
            for j in range(len(test_all_token_ids_list[i])):
                # 遍历每个样本中的每个分段文本
                if self.add_soft_embeddings:
                    cur_seg_soft_embeddings = test_all_soft_embeddings[i][j]
                else:
                    cur_seg_soft_embeddings = None
                cur_seg_start_probs, cur_seg_end_probs = self.predict_lear(test_all_token_ids_list[i][j],
                                                                           test_all_token_type_ids_list[i][j],
                                                                           test_all_input_masks_list[i][j],
                                                                           test_all_input_lens_list[i][j],
                                                                           cur_seg_soft_embeddings,
                                                                           label_tokens_ids_list,
                                                                           label_token_type_ids_list,
                                                                           label_token_masks_list,
                                                                           label_tokens_length_list,
                                                                           label_soft_embeddings)
                cur_seg_result = self.extract_span_from_start_end(cur_seg_start_probs, cur_seg_end_probs,
                                                                  test_all_token_ids_list[i][j],
                                                                  test_all_input_lens_list[i][j])
                cur_sample_result_list.extend(cur_seg_result)
            cur_sample_result_list = list(set(cur_sample_result_list))
            prediction_result_list.append(cur_sample_result_list)
        end = datetime.datetime.now()
        print("time_span:{}".format((end - begin).seconds))
        return prediction_result_list

    def eval_prediction_with_gold_label(self, orig_test_file):
        orig_texts, orig_labels = gen_orig_test_text_label(orig_test_file, False)
        id2slot_dict = self.data_loader.tokenizer.id2slot
        true_entity_list = gen_entity_from_label_id_list(orig_texts, orig_labels, id2slot_dict, orig_test=True)
        prediction_entity_list = self.predict_for_all_data(orig_test_file)
        cal_mertric_from_two_list(prediction_entity_list, true_entity_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_result_path", default='prediction_result_baseline.npy', type=str)
    parser.add_argument("--model_pb_dir", default='lear_model_pb', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--label_less", action='store_true', default=False)
    parser.add_argument("--has_cls", action='store_true', default=False)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--pre_buffer_size", default=1, type=int)
    parser.add_argument("--add_soft_embeddings", default=False, action='store_true')

    # parser.add_argument("--model_pb_dir", default='base_pb_model_dir', type=str)
    args = parser.parse_args()
    fp = fastPredict(lear_config.get(args.model_pb_dir), args.add_soft_embeddings, lear_config)
    # fp.predict_for_all_data(os.path.join(lear_config.get("data_dir"),lear_config.get("orig_test")))
    fp.eval_prediction_with_gold_label(os.path.join(lear_config.get("data_dir"), lear_config.get("orig_test")))
