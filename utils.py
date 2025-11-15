import numpy as np
from json import dump

def pre(text : str):
    text = text.strip().lower()
    text = "".join(text.split())
    return text

def post_process(result):    
    texts = np.array(result['rec_texts'])
    key = ['date', 'family name', 'first name', 'candidate id', 'date of birth', 'sex (m/f)', 'band', 'date']
    iterater = iter(key)
    k = next(iterater)

    out = {}
    for ind in range(len(texts)):
        # print(pre(texts[ind]))
        # print(pre(k))
        if pre(k) in pre(texts[ind]):
            # print(texts[ind+1])
            out[k] = str(texts[ind+1])
            k = next(iterater, None)
            if k is None:
                break
    return out

def save_output(out, filename="output.json"):
    with open(filename, "w") as f:
        dump(out, f)
