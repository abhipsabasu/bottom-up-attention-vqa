import json
from os.path import join
import random
import pickle

random.seed(0)

question_path = join(
    'data', 'vqacp_v2_train_questions.json')
with open(question_path) as f:
    questions = json.load(f)
questions.sort(key=lambda x: x['question_id'])

answer_path = join('data', 'cp-cache', 'train_target.pkl')
with open(answer_path, 'rb') as f:
    answers = pickle.load(f)
answers.sort(key=lambda x: x['question_id'])
print(answers[0].keys())

question_id = [q['question_id'] for q in questions]
image_id = [q['image_id'] for q in questions]
# dataset = []
shuffled_dataset = []

print(questions[0].keys())

for i in range(len(questions)):
    while True:
        new = random.randint(0, len(question_id))
        if image_id[i] != image_id[new]:
            break
    if questions[i]['question_id'] != question_id[i] or question_id[i] != answers[i]['question_id']:
        print('not same')
        break
    data = {'question_id': question_id[i], 'question': questions[i]['question'], 'image_id': image_id[i],
            'question_type': answers[i]['question_type'], 'label': 1}
    shuffled_data = {'question_id': question_id[i], 'question': questions[i]['question'], 'image_id': image_id[new],
                     'question_type': answers[i]['question_type'], 'label': 0}
    shuffled_dataset.append(shuffled_data)
    shuffled_dataset.append(data)
    # dataset.append(data)

with open(join('data', "shuffled_data.json"), "w") as f:
    json.dump(shuffled_dataset, f, indent=2)
