import codecs
import requests
import json
import jieba
SEGURL=""

def seg_from_api(data):
    try:
        datas = {"text": data}
        headers = {'Content-Type': 'application/json'}
        res = requests.post(SEGURL, data=json.dumps(datas), headers=headers)
        text = res.text
        text_dict = json.loads(text)
        return text_dict
    except:
        print("dfdfdf")

def seg_from_jieba(data):
    seg_list = jieba.cut(data, cut_all=True)
    return seg_list

def seg(text,jieba=True):
    """
    可以使用jieba分词，也可以用已有的分词api
    :param text:
    :return:
    """
    if not jieba:
        words = seg_from_api(text)
        word_list = [word.get("word") for word in words]
        return word_list
    else:
        return words

def read_slots(slot_file_path=None,slot_source_type="file"):
    """
    根据不同的槽位模板文件，生成槽位的label
    :param slot_file_path:
    :param slot_source_type:
    :return:
    """
    slot2id_dict = {}
    id2slot_dict = {}
    if slot_source_type == "file":
        with codecs.open(slot_file_path,'r','utf-8') as fr:
            for i,line in enumerate(fr):
                line=line.strip("\n")
                line = line.strip("\r")
                slot2id_dict[line] = i
                id2slot_dict[i] = line
    return slot2id_dict,id2slot_dict