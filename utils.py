import numpy as np
from json import dump
import re

def pre(text: str):
    text = text.strip().lower()
    text = "".join(text.split())
    text = re.sub(r'[^a-z]', '', text)  # giữ lại chỉ chữ cái
    return text

def post_process(texts):    
    key = ['date', 'family name', 'first name', 'candidate id', 'date of birth', 'sex (m/f)', 'band', 'date end']
    iterater = iter(key)
    k = next(iterater)

    out = {}
    for ind in range(len(texts)):
        # print(pre(texts[ind]))
        # print(pre(k))
        key = pre(k)
        text = pre(texts[ind])
        if abs(len(text) - len(key)) >= 10:
            continue
        if key in text or is_equivalent(key, text):
            # print(texts[ind+1])
            out[k] = str(texts[ind+1])
            k = next(iterater, None)
            if k is None:
                break
    for ind in range(len(texts) - 1, -1, -1):
        if 'date' in pre(texts[ind]):
            out['date end'] = str(texts[ind+1])
            break
    return out

def save_output(out, filename="output.json"):
    with open(filename, "w") as f:
        dump(out, f)
def similarity_ratio(s1: str, s2: str):
    if len(s1) != len(s2):
        return 0.0

    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    return matches / len(s1)


def is_equivalent(s1: str, s2: str, threshold=0.6):
    return similarity_ratio(s1, s2) >= threshold