import pickle
import json
import os
import random

random.seed(0)

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

name = 'train'
dataroot = 'data'
question_path = os.path.join(
          dataroot, 'vqacp_v2_%s_questions.json' % name)
with open(question_path) as f:
    questions = json.load(f)
questions.sort(key=lambda x: x['question_id'])
qids = {questions[i]['question_id']:i for i in range(len(questions))}
print(len(questions), len(qids.keys()))

question2_path = os.path.join(
          dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
with open(question2_path) as f:
    vqa2questions = json.load(f)["questions"]

answer_path = os.path.join(dataroot, 'cp-cache', '%s_target.pkl' % name)
with open(answer_path, 'rb') as f:
    answers = pickle.load(f)
answers.sort(key=lambda x: x['question_id'])
print(len(answers))

assert_eq(len(answers), len(questions))

val_index = [] #
rand_idx = random.sample(range(0, len(vqa2questions)), len(vqa2questions))

for idx in rand_idx:
    q_id = vqa2questions[idx]['question_id']
    if q_id in qids:
        # print(q_id, len(val_index))
        val_index.append(qids[q_id])
        if len(val_index) == int(0.1 * len(questions)):
            break

index = [i for i in range(len(questions))]  # if i not in val_index]
val_index.sort()
train_index = list(set(index) - set(val_index))
train_index.sort()


val_questions = [questions[i] for i in val_index]
train_questions = [questions[i] for i in train_index]

val_answers = [answers[i] for i in val_index]
train_answers = [answers[i] for i in train_index]

with open(os.path.join(dataroot, 'vqacp_v2_traindev_questions.json'), "w") as f:
    json.dump(train_questions, f, indent=2)

with open(os.path.join(dataroot, 'vqacp_v2_dev_questions.json'), "w") as f:
    json.dump(val_questions, f, indent=2)

pickle.dump(val_answers, open(os.path.join(dataroot, 'cp-cache', 'dev_target.pkl'), 'wb'))
pickle.dump(train_answers, open(os.path.join(dataroot, 'cp-cache', 'traindev_target.pkl'), 'wb'))