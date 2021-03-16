# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# @Author: Ryuchen
# @Time: 2021/03/1-08:32
# @Site: https://ryuchen.github.io
# @Contact: chenhaom1993@hotmail.com
# @Copyright: Copyright (C) 2019-2020 Bistu-DZ.
# ========================================================
"""
...
DocString Here
...
"""

import codecs

from utils import data_utils


def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是以空格进行分割
    最后返回句子集合
    :param path:
    :return:
    """
    # 存放数据集
    sentences = []
    # 临时存放每一个句子
    sentence = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        # 逐行去读取数据集内容，去掉两边空格
        line = line.strip()
        # 前一步把空格去掉了，所以这一步就变成了空字符串
        # 首先判断是不是空，如果是则表示句子和句子之间的分割点
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                # 清空sentence表示一句话完结
                sentence = []
        # 如果不是，就将当前的字和标记保存下来，放到临时存放的地方
        else:
            if line[0] == " ":
                continue
            else:
                word = line.split()
                assert len(word) >= 2
                sentence.append(word)
    # 循环走完，要判断一下，防止最后一个句子没有进入到句子集合中
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    更新为指定编码 BIOES 限定了输入的句子编码只能为BIO编码
    :param sentences:
    :param tag_scheme:
    :return:
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if not data_utils.check_bio(tags):
            s_str = "\n".join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子%i:\n%s" % (i, s_str))

        if tag_scheme == "BIO":
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag

        if tag_scheme == "BIOES":
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception("非法目标编码")


def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[x[0] for x in s] for s in sentences]
    dico = data_utils.create_dico(word_list)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = data_utils.create_mapping(dico)
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences:
    :return:
    """
    tag_list = [[x[1] for x in s] for s in sentences]
    dico = data_utils.create_dico(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(dico)
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，返回list其实包含
    -word_list
    -word_id_list
    -word char indexs
    -tag_id_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    none_index = tag_to_id['O']

    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        segs = data_utils.get_seg_features("".join(word_list))
        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            tag_id_list = [none_index for w in s]
        data.append([word_list, word_id_list, segs, tag_id_list])

    return data


if __name__ == "__main__":
    path = "../data/NER/ner.dev"
    sentences = load_sentences(path)
    update_tag_scheme(sentences, "BIOES")
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    # 返回的是列表，列表中的每个元素也是一个列表四个元素的列表包含：
    # 0：句子字的列表-> word_list ['相', '比', '之', '下']
    # 1：句子字的id列表 -> word_id_list [240, 181, 127, 70]
    # 2：句子字的分词列表 -> segs_list [1, 2, 2, 3] === 0 表示单个字成词, 1 表示一个词的开始， 2 表示一个词的中间，3 表示一个词的结尾
    # 3：句子字的bioes标记列表 -> segs_list [0, 0, 0, 0]
    dev_data = prepare_dataset(sentences, word_to_id, tag_to_id)
    data_utils.BatchManager(dev_data, 120)
