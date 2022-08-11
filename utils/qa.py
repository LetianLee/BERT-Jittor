import jittor as jt
import json
from tqdm import tqdm


class QA:

    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, context, question):
        # Tokenization
        inputs = self.tokenizer(context, question, truncation=True, max_length=512)
        for key in inputs.keys():
            inputs[key] = jt.array(inputs[key]).unsqueeze(0)

        # Model makes its prediction
        outputs = self.model(**inputs)
        prediction = self.__predict__(inputs, outputs)
        return prediction

    # Prediction function
    def __predict__(self, inputs, outputs):
        answer_start_index = jt.argmax(outputs["start_logits"], dim=1)[0]
        answer_end_index = jt.argmax(outputs["end_logits"], dim=1)[0]
        predict_answer_tokens = [
            inputs["input_ids"][i][int(answer_start_index[i]): int(answer_end_index[i]) + 1].numpy() for i in
            range(len(inputs["input_ids"]))]
        predictions = [self.tokenizer.decode(tokens) for tokens in predict_answer_tokens]
        return predictions[0]


class InferQA:

    def __init__(self, model, tokenizer, data_path):
        self.QA_model = QA(model, tokenizer)

        with open(data_path) as f:
            data = json.load(f)["data"]
        self.data = data

    def infer(self):
        f1 = exact_match = total = 0
        pbar = tqdm(self.data)
        for d in pbar:
            title = d['title']
            for p in d['paragraphs']:
                c = p['context']
                for qa in p['qas']:
                    q = qa["question"]
                    a = [an['text'] for an in qa['answers']]
                    total += 1
                    pred = self.QA_model.predict(c, q)
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, pred, a)
                    f1 += metric_max_over_ground_truths(
                        f1_score, pred, a)
                    pbar.set_postfix({
                        'total': total,
                        'match': '{:.3f}'.format(exact_match / total),
                        'f1': '{:.3f}'.format(f1 / total)
                    })
        pbar.close()
        print("exact_match: %s, f1: %s" % ('{:.3f}'.format(exact_match / total), '{:.3f}'.format(f1 / total)))


""" Official evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
