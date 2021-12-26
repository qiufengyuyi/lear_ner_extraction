import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import time
import logging
from common_utils import set_logger
import tensorflow as tf
from sklearn.metrics import f1_score
from models.model_lear import lear_model_fn_builder
from data_processing.lear_processor import LEARPrepareData, input_lear_fn
from configs.lear_config import lear_config
from sklearn.metrics import classification_report, f1_score

logger = set_logger("[run training]")


def lear_serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    features['words'],features["label_tokens"],features["label_token_type_ids"],features['token_type_ids'],
    features['text_length'],features['label_text_length'],features['token_masks'],features['label_token_masks']
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.int32, shape=[None, None], name='words')
    label_tokens = tf.placeholder(dtype=tf.int32, shape=[3, 24], name="label_tokens")
    label_token_type_ids = tf.placeholder(dtype=tf.int32, shape=[3, 24], name="label_token_type_ids")
    token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='token_type_ids')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='text_length')
    label_text_length = tf.placeholder(dtype=tf.int32, shape=[3], name='label_text_length')
    token_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='token_masks')
    label_token_masks = tf.placeholder(dtype=tf.int32, shape=[3, 24], name='label_token_masks')
    input_lexicons = tf.placeholder(dtype=tf.float32, shape=[None, None, 1200], name='input_lexicons')
    label_lexicons = tf.placeholder(dtype=tf.float32, shape=[3, 24, 1200], name='label_lexicons')
    receiver_tensors = {'words': words, 'label_tokens': label_tokens, 'label_token_type_ids': label_token_type_ids,
                        'token_type_ids': token_type_ids, 'text_length': nwords, 'label_text_length': label_text_length,
                        'token_masks': token_masks, 'label_token_masks': label_token_masks,
                        "input_lexicons": input_lexicons, "label_lexicons": label_lexicons}
    features = {'words': words, 'label_tokens': label_tokens, 'label_token_type_ids': label_token_type_ids,
                'token_type_ids': token_type_ids, 'text_length': nwords, 'label_text_length': label_text_length,
                'token_masks': token_masks, 'label_token_masks': label_token_masks, "input_lexicons": input_lexicons,
                "label_lexicons": label_lexicons}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def _f1_bigger(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

    Both evaluation results should have the values for MetricKeys.LOSS, which are
    used for comparison.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.

    Raises:
      ValueError: If input eval result is None or no loss is available.
    """
    default_key = "f1_start_micro"
    default_key_2 = "f1_end_micro"
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] + best_eval_result[default_key_2] < current_eval_result[default_key] + \
           current_eval_result[default_key_2]


def run_lear(args):
    # hvd.init()
    vocab_file_path = os.path.join(lear_config.get("bert_pretrained_model_path"), lear_config.get("vocab_file"))
    bert_config_file = os.path.join(lear_config.get("bert_pretrained_model_path"), lear_config.get("bert_config_path"))
    slot_file = os.path.join(lear_config.get("slot_list_root_path"), lear_config.get("bert_slot_file_name"))
    data_loader = LEARPrepareData(vocab_file_path, slot_file, lear_config, bert_config_file, 384, True, False, args.add_soft_embeddings,
                                  "data/soft_vocab_count.json", "data/bio_word2vec_trim", "data/gaz_position_dict.json")
    label_tokens_ids_list, label_token_type_ids_list, label_token_masks_list, label_tokens_length_list, label_soft_embeddings_list = data_loader.gen_label_tokens()
    train_data_X, train_data_start_Y, train_data_end_Y, train_token_type_ids_list, train_input_masks_list, train_soft_embeddings_list = data_loader.gen_train_data_from_raw(
        "train")
    dev_data_X, dev_data_start_Y, dev_data_end_Y, dev_token_type_ids_list, dev_input_masks_list, dev_soft_embeddings_list = data_loader.gen_train_data_from_raw(
        "dev")
    # print(data_loader.train_valid_split_data_path)
    train_samples_nums = len(train_data_X)
    # np.save("data/train_soft_embeddings.npy",train_soft_embeddings_list)
    # np.save("data/dev_soft_embeddings.npy",dev_soft_embeddings_list)
    if train_samples_nums % args.train_batch_size != 0:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size) + 1
    else:
        each_epoch_steps = int(train_samples_nums / args.train_batch_size)
    # each_epoch_steps = int(data_loader.train_samples_nums/args.train_batch_size)+1
    logger.info('*****train_set sample nums:{}'.format(train_samples_nums))
    logger.info('*****train each epoch steps:{}'.format(each_epoch_steps))
    train_steps_nums = each_epoch_steps * args.epochs
    # train_steps_nums = each_epoch_steps * args.epochs // hvd.size()
    logger.info('*****train_total_steps:{}'.format(train_steps_nums))
    decay_steps = args.decay_epoch * each_epoch_steps
    logger.info('*****train decay steps:{}'.format(decay_steps))
    # dropout_prob是丢弃概率
    params = {"dropout_prob": args.dropout_prob, "num_labels": 3,
              "rnn_size": 256, "num_layers": args.num_layers, "hidden_units": args.hidden_units,
              "decay_steps": decay_steps, "add_soft_embeddings": args.add_soft_embeddings}
    # dist_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpu_nums)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    # "bert_ce_model_dir"
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # config_tf.gpu_options.visible_device_list = str(hvd.local_rank())
    # checkpoint_path = os.path.join(bert_config.get(args.model_checkpoint_dir), str(hvd.rank()))
    run_config = tf.estimator.RunConfig(
        model_dir=lear_config.get(args.model_checkpoint_dir),
        save_summary_steps=train_steps_nums,
        save_checkpoints_steps=each_epoch_steps,
        session_config=config_tf,
        keep_checkpoint_max=1,
        # train_distribute=dist_strategy
    )

    bert_init_checkpoints = os.path.join(lear_config.get("bert_pretrained_model_path"),
                                         lear_config.get("bert_init_checkpoints"))
    model_fn = lear_model_fn_builder(bert_config_file, bert_init_checkpoints, args)
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)
    # train_hook_one = RestoreCheckpointHook(bert_init_checkpoints)
    # early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
    #     estimator=estimator,
    #     metric_name='loss',
    #     max_steps_without_decrease=args.tolerant_steps,
    #     eval_dir=None,
    #     min_steps=0,
    #     run_every_secs=None,
    #     run_every_steps=args.run_hook_steps)
    if args.do_train:
        # label_tokens_ids_list,label_token_type_ids_list,label_token_masks_list,label_tokens_length_list
        # input_Xs,input_label_tokens,input_token_type_ids,label_token_type_ids,input_masks,label_masks,start_Ys,end_Ys,label_token_lens,is_training,is_testing,args
        # train_data_X,train_data_start_Y,train_data_end_Y,train_token_type_ids_list,train_input_masks_list
        # input_Xs,input_label_tokens,input_token_type_ids,label_token_type_ids,input_masks,label_masks,start_Ys,end_Ys,label_token_lens,is_training,is_testing,args
        train_input_fn = lambda: input_lear_fn(input_Xs=train_data_X, input_label_tokens=label_tokens_ids_list,
                                               input_token_type_ids=train_token_type_ids_list,
                                               label_token_type_ids=label_token_type_ids_list,
                                               input_masks=train_input_masks_list, label_masks=label_token_masks_list,
                                               start_Ys=train_data_start_Y, end_Ys=train_data_end_Y,
                                               label_token_lens=label_tokens_length_list,
                                               input_lexicons_list=train_soft_embeddings_list,
                                               label_lexicons=label_soft_embeddings_list, is_training=True,
                                               is_testing=False, args=args)

        eval_input_fn = lambda: input_lear_fn(dev_data_X, label_tokens_ids_list,
                                              dev_token_type_ids_list, label_token_type_ids_list, dev_input_masks_list,
                                              label_token_masks_list, dev_data_start_Y, dev_data_end_Y,
                                              label_tokens_length_list, input_lexicons_list=dev_soft_embeddings_list,
                                              label_lexicons=label_soft_embeddings_list, is_training=False,
                                              is_testing=False, args=args)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps_nums
                                            )
        exporter = tf.estimator.BestExporter(exports_to_keep=1,
                                             serving_input_receiver_fn=lear_serving_input_receiver_fn,
                                             compare_fn=_f1_bigger)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter], throttle_secs=0)
        # for _ in range(args.epochs):

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        # estimator.train(train_input_fn,max_steps=train_steps_nums)
        # "bert_ce_model_pb"
        estimator.export_saved_model(lear_config.get(args.model_pb_dir), lear_serving_input_receiver_fn)
