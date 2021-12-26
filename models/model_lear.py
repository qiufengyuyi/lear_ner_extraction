import tensorflow as tf
import math
import common_utils
import optimization
from models.tf_metrics import f1
# from albert import modeling,modeling_google
from bert import modeling
# from bert import modeling_theseus
# from models.utils import focal_loss
# from tensorflow.python.ops import metrics as metrics_lib

logger = common_utils.set_logger('NER Training...')


def cal_bin_loss(logits, labels, input_masks):
    # input_masks = tf.cast(tf.reshape(input_masks,(-1,1,1)),dtype=tf.float32)
    # logits *= input_masks
    # labels *= input_masks
    labels = tf.cast(labels, dtype=tf.float32)
    cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    # 目前是所有类别都是一个权重
    cur_loss = tf.reduce_sum(cur_loss, axis=-1)
    input_masks = tf.cast(input_masks, dtype=tf.float32)
    cur_loss = cur_loss * input_masks
    cur_loss = tf.reduce_sum(cur_loss, axis=-1)
    cur_loss = tf.reduce_mean(cur_loss)
    return cur_loss


class LEAR(object):
    def __init__(self, params, bert_config):
        # 丢弃概率
        self.dropout_rate = params["dropout_prob"]
        self.num_labels = params["num_labels"]
        self.add_soft_embeddings = params["add_soft_embeddings"]
        self.bert_config = bert_config
        if self.add_soft_embeddings:
            self.attention_size = self.bert_config.hidden_size + 1200
        else:
            self.attention_size = self.bert_config.hidden_size

    def fusion_seq_with_label_tokens(self, token_features, label_features, token_masks, label_token_masks):
        # batch_size,seq_len,hidden_size
        token_features_fc = tf.layers.dense(token_features, self.attention_size, use_bias=False,
                                            kernel_initializer=modeling.create_initializer(
                                                self.bert_config.initializer_range))
        input_shape = modeling.get_shape_list(token_features_fc)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        label_features_fc = tf.layers.dense(label_features, self.attention_size, use_bias=False,
                                            kernel_initializer=modeling.create_initializer(
                                                self.bert_config.initializer_range))
        # hidden_size,label_num,label_seq_len
        label_features_t = tf.transpose(label_features_fc, [2, 0, 1])
        # hidden_size,label_num*label_seq_len
        label_features_t = tf.reshape(label_features_t, (self.attention_size, -1))
        # batch_size,seq_len,label_num*label_seq_len,
        atten_scores = tf.matmul(token_features_fc, label_features_t)
        # atten_scores = tf.multiply(atten_scores,
        #                            1.0 / math.sqrt(float(self.attention_size)))
        # atten_scores = atten_scores / math.sqrt(self.hidden_size)
        # batch_size,seq_len,label_num,label_seq_len
        atten_scores = tf.reshape(atten_scores, (batch_size, seq_length, self.num_labels, -1))
        # 1,1,label_num,label_seq_len
        label_token_masks = label_token_masks[None, None, :, :]
        label_token_masks = (1.0 - tf.cast(label_token_masks, tf.float32)) * -10000.0
        atten_scores += label_token_masks
        # # try add context token masks
        # #batch_size,seq_len,1,1
        # token_masks = [:,:,None,None]
        # token_masks = (1.0 - tf.cast(token_masks, tf.float32)) * -10000.0
        # atten_scores += token_masks
        attention_probs = tf.nn.softmax(atten_scores)
        # batch_size,seq_len,label_num,label_seq_len,1
        attention_probs = tf.expand_dims(attention_probs, axis=-1)

        value_layers = tf.expand_dims(label_features_fc, axis=0)
        value_layers = tf.expand_dims(value_layers, axis=0)
        # batch_size,seq_len,label_num,label_seq_len,hidden_size
        value_layers = tf.tile(value_layers, [batch_size, seq_length, 1, 1, 1])
        # batch_size,seq_len,label_num,label_seq_len,hidden_size
        context_label_features = value_layers * attention_probs
        # batch_size,seq_len,label_num,hidden_size
        weighted_context_label_features_sum = tf.reduce_sum(context_label_features, axis=-2)
        token_features_fc = tf.expand_dims(token_features, axis=2)
        # concat
        # token_features_fc = tf.tile(token_features_fc,[1,1,self.num_labels,1])
        # fused_feature = tf.concat([token_features_fc,weighted_context_label_features_sum],axis=-1)
        fused_feature = token_features_fc + weighted_context_label_features_sum
        output = tf.layers.dense(fused_feature, self.attention_size, activation=tf.nn.tanh,
                                 kernel_initializer=modeling.create_initializer(self.bert_config.initializer_range))
        # [batch_size, input_seq_len, num_labels, hidden_size]
        return output

    def classifier_layers(self, input_features, type="start"):
        # input features
        # batch_size,input_seq_len,num_labels,hidden_size
        # input_shape = modeling.get_shape_list(input_features)
        # batch_size = input_shape[0]
        # seq_length = input_shape[1]
        classifer_weight = tf.get_variable(
            name="classifier_weight" + "_" + type,
            shape=[self.num_labels, self.attention_size],
            initializer=modeling.create_initializer(self.bert_config.initializer_range))
        classifer_bias = tf.get_variable(name="classifier_bias" + "_" + type, shape=[self.attention_size],
                                         initializer=tf.constant_initializer(0))
        output = tf.multiply(input_features, classifer_weight)
        output += classifer_bias
        # [batch_size, input_seq_len, num_labels]
        output = tf.reduce_sum(output, axis=-1)
        return output

    def __call__(self, input_ids, label_token_ids, label_token_type_ids, token_type_ids_list, text_length_list,
                 label_token_length, token_masks, label_token_masks, start_labels=None, end_labels=None,
                 is_training=True, is_testing=False, input_lexicons=None, label_lexicons=None):
        if label_token_ids.shape.ndims == 3:
            label_token_ids = label_token_ids[0, :, :]
            label_token_type_ids = label_token_type_ids[0, :, :]
            label_token_masks = label_token_masks[0, :, :]
            label_token_length = label_token_length[0, :]
            label_lexicons = label_lexicons[0, :, :, :]
        # print(label_token_i =ds.shape.ndims)
        bert_model_1 = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            input_mask=token_masks,
            token_type_ids=token_type_ids_list,
            use_one_hot_embeddings=False, scope="bert"
        )
        bert_seq_output = bert_model_1.get_sequence_output()
        if self.add_soft_embeddings:
            bert_seq_output = tf.concat([bert_seq_output, input_lexicons], axis=-1)

        bert_model_2 = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=label_token_ids,
            text_length=label_token_length,
            input_mask=label_token_masks,
            token_type_ids=label_token_type_ids,
            use_one_hot_embeddings=False, scope="bert"
        )

        label_seq_output = bert_model_2.get_sequence_output()
        if self.add_soft_embeddings:
            label_seq_output = tf.concat([label_seq_output, label_lexicons], axis=-1)
        # bert_project = tf.layers.dense(bert_seq_output, self.hidden_units, activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project, rate=self.dropout_rate, training=is_training)
        fused_output = self.fusion_seq_with_label_tokens(bert_seq_output, label_seq_output, token_masks,
                                                         label_token_masks)
        # [batch_size, input_seq_len, num_labels]
        start_logits = self.classifier_layers(fused_output, "start")
        end_logits = self.classifier_layers(fused_output, "end")
        predict_start_probs = tf.sigmoid(start_logits)

        predict_end_probs = tf.sigmoid(end_logits)
        if not is_testing:
            # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
            # start_loss = ce_loss(start_logits,start_labels,final_mask,self.num_labels,True)
            # end_loss = ce_loss(end_logits,end_labels,final_mask,self.num_labels,True)

            # focal loss
            # start_loss = focal_loss(
            #     start_logits, start_labels, final_mask, self.num_labels, True, 2)
            # end_loss = focal_loss(end_logits, end_labels,
            #                       final_mask, self.num_labels, True, 2)
            start_loss = cal_bin_loss(start_logits, start_labels, token_masks)
            end_loss = cal_bin_loss(end_logits, end_labels, token_masks)
            final_loss = start_loss + end_loss
            return final_loss, predict_start_probs, predict_end_probs
        else:
            return predict_start_probs, predict_end_probs


def lear_model_fn_builder(bert_config_file, init_checkpoints, args):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        if isinstance(features, dict):
            # input_ids, label_token_ids, label_token_type_ids, token_type_ids_list, text_length_list, label_token_length, token_masks, label_token_masks, start_labels, end_labels,
            features = features['words'], features["label_tokens"], features["label_token_type_ids"], features[
                'token_type_ids'], features['text_length'], features['label_text_length'], features['token_masks'], \
                       features['label_token_masks'], features["input_lexicons"], features["label_lexicons"]
        print(features)
        input_ids, label_token_ids, label_token_type_ids, token_type_ids_list, text_length_list, label_token_length, token_masks, label_token_masks, input_lexicons, label_lexicons = features
        if labels is not None:
            start_labels, end_labels = labels
        else:
            start_labels, end_labels = None, None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_testing = (mode == tf.estimator.ModeKeys.PREDICT)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tag_model = LEAR(params, bert_config)
        # tag_model = bertMRCMultiClass(params,bert_config)
        # input_ids,labels,token_type_ids_list,query_len_list,text_length_list,is_training,is_testing=False
        if is_testing:
            predict_start_probs, predict_end_probs = tag_model(input_ids, label_token_ids, label_token_type_ids,
                                                               token_type_ids_list, text_length_list,
                                                               label_token_length, token_masks, label_token_masks, None,
                                                               None, is_training, is_testing,
                                                               input_lexicons=input_lexicons,
                                                               label_lexicons=label_lexicons)
            # predict_ids,weight,predict_prob = tag_model(input_ids,labels,token_type_id_list,query_length_list,text_length_list,is_training,is_testing)
        else:
            loss, predict_start_probs, predict_end_probs = tag_model(input_ids, label_token_ids, label_token_type_ids,
                                                                     token_type_ids_list, text_length_list,
                                                                     label_token_length, token_masks, label_token_masks,
                                                                     start_labels, end_labels, is_training,
                                                                     input_lexicons=input_lexicons,
                                                                     label_lexicons=label_lexicons)
            # loss,predict_ids,weight,predict_prob = tag_model(input_ids,labels,token_type_id_list,query_length_list,text_length_list,is_training,is_testing)

        # def metric_fn(label_ids, pred_ids):
        #     return {
        #         'precision': precision(label_ids, pred_ids, params["num_labels"]),
        #         'recall': recall(label_ids, pred_ids, params["num_labels"]),
        #         'f1': f1(label_ids, pred_ids, params["num_labels"])
        #     }
        #
        # eval_metrics = metric_fn(labels, pred_ids)
        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoints:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoints)
            tf.train.init_from_checkpoint(init_checkpoints, assignment_map)
        output_spec = None
        # f1_score_val, f1_update_op_val = f1(labels=labels, predictions=pred_ids, num_classes=params["num_labels"],
        #                                     weights=weight)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, args.lr, args.clip_norm)
            hook_dict = {}
            # precision_score, precision_update_op = precision(labels=labels, predictions=pred_ids,
            #                                                  num_classes=params["num_labels"], weights=weight)
            #
            # recall_score, recall_update_op = recall(labels=labels,
            #                                         predictions=pred_ids, num_classes=params["num_labels"],
            #                                         weights=weight)
            hook_dict['loss'] = loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.print_log_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            start_labels_list = tf.split(start_labels, 3, axis=-1)
            start_labels_con = tf.concat(start_labels_list, axis=0)
            end_labels_list = tf.split(end_labels, 3, axis=-1)
            end_labels_con = tf.concat(end_labels_list, axis=0)
            pred_start_probs_list = tf.split(predict_start_probs, 3, axis=-1)
            pred_start_probs_con = tf.concat(pred_start_probs_list, axis=0)
            pred_end_probs_list = tf.split(predict_end_probs, 3, axis=-1)
            pred_end_probs_con = tf.concat(pred_end_probs_list, axis=0)
            pred_start_ids = tf.greater(pred_start_probs_con, 0.5)
            pred_end_ids = tf.greater(pred_end_probs_con, 0.5)
            weight = tf.concat([token_masks, token_masks, token_masks], axis=0)
            # pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # weight = tf.sequence_mask(text_length_list)
            # precision_score, precision_update_op = precision(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            #
            # recall_score, recall_update_op =recall(labels=labels,
            #                                              predictions=pred_ids,num_classes=params["num_labels"],weights=weight)
            # def metric_fn(per_example_loss, label_ids, probabilities):

            #     logits_split = tf.split(probabilities, params["num_labels"], axis=-1)
            #     label_ids_split = tf.split(label_ids, params["num_labels"], axis=-1)
            #     # metrics change to auc of every class
            #     eval_dict = {}
            #     for j, logits in enumerate(logits_split):
            #         label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
            #         current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
            #         eval_dict[str(j)] = (current_auc, update_op_auc)
            #     eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
            #     return eval_dict
            # eval_metrics = metric_fn(per_example_loss, labels, pred_ids)
            f1_start_val, f1_update_op_val = f1(labels=start_labels_con, predictions=pred_start_ids, num_classes=2,
                                                weights=weight, average="micro")
            f1_end_val, f1_end_update_op_val = f1(labels=end_labels_con, predictions=pred_end_ids, num_classes=2,
                                                  weights=weight, average="micro")
            # f1_val,f1_update_op_val = f1(labels=labels,predictions=predict_ids,num_classes=3,weights=weight,average="macro")

            # f1_score_val_micro,f1_update_op_val_micro = f1(labels=labels,predictions=pred_ids,num_classes=params["num_labels"],weights=weight,average="micro")

            # acc_score_val,acc_score_op_val = tf.metrics.accuracy(labels=labels,predictions=pred_ids,weights=weight)
            # eval_loss = tf.metrics.mean_squared_error(labels=labels, predictions=pred_ids,weights=weight)

            eval_metric_ops = {
                "f1_start_micro": (f1_start_val, f1_update_op_val),
                "f1_end_micro": (f1_end_val, f1_end_update_op_val),
                "eval_loss": tf.metrics.mean(values=loss)}

            # eval_metric_ops = {
            # "f1_macro":(f1_val,f1_update_op_val),
            # "eval_loss":tf.metrics.mean(values=loss)}

            # eval_hook_dict = {"f1":f1_score_val,"loss":loss}

            # eval_logging_hook = tf.train.LoggingTensorHook(
            #     at_end=True,every_n_iter=args.print_log_steps)
            output_spec = tf.estimator.EstimatorSpec(
                eval_metric_ops=eval_metric_ops,
                mode=mode,
                loss=loss
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"start_probs": predict_start_probs, "end_probs": predict_end_probs}
            )
            # output_spec = tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     predictions={"pred_ids":predict_ids,"pred_probs":predict_prob}
            # )
        return output_spec

    return model_fn
