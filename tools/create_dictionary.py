from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json',
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    files = [
        'vqacp_v2_test_annotations.json',
        'vqacp_v2_train_annotations.json'
    ]
    for path in files:
        ans_path = os.path.join(dataroot, path)
        ans_json = json.load(open(ans_path))
        for dic in ans_json:
            mca = dic['multiple_choice_answer']
            dictionary.tokenize(mca, True)
            for ans in dic['answers']:
                dictionary.tokenize(ans['answer'], True)
    print(dictionary.word2idx)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(x) for x in vals[1:]]#map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary('data')
    d.dump_to_file('data/dictionary.pkl')

    # d = Dictionary.load_from_file('data/dictionary.pkl')
    # emb_dim = 300
    # glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    # weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    # np.save('data/glove6b_init_%dd.npy' % emb_dim, weights)
