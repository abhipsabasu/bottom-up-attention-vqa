import json
from os.path import join
import random
import pickle

random.seed(0)

question_path = join(
    'data', 'vqacp_v2_test_questions.json')
with open(question_path) as f:
    questions = json.load(f)
questions.sort(key=lambda x: x['question_id'])

question_id = [q['question_id'] for q in questions]
image_id = [q['image_id'] for q in questions]
qlen = len(question_id)
print(question_id[:10])

random.shuffle(question_id)

q_one = []
q_two = []
im_one = set()

for i in range(qlen):
    print(i, qlen)
    if i < qlen//2:
        q_one.append(question_id[i])
        im_one.add(image_id[i])
    elif i == qlen // 2:
        j = i
        while image_id[j] in im_one:
            q_one.append(question_id[j])
            im_one.add(image_id[j])
            j = j + 1
        break
print("Reached")
q_two = list(set(question_id) - set(q_one))
q_one = list(set(q_one))
print(len(q_two), len(set(q_two)))
print(len(q_one),len(set(q_one)))

with open('data/test_train.pkl', 'wb') as fp:
    pickle.dump(q_one, fp)

with open('data/test.pkl', 'wb') as fp:
    pickle.dump(q_two, fp)


