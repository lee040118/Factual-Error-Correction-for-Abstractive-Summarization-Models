import argparse
import json, re
import os
import re
import pickle
import random
from tqdm import tqdm
from pororo import Pororo
from collections import defaultdict
from googletrans import Translator

LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}

def load_json(file_path):
    return json.load(open(file_path, 'r'))

# def load_json(file_path):
#     with open(file_path, encoding="utf-8") as f:
#         data = [json.loads(line) for line in f]
#     return data

def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token

def save_data(args, data, name_suffix=None):
    if name_suffix:
        output_file = os.path.splitext(args.source_file)[0] + "-" + name_suffix +  ".jsonl"
    else:
        output_file = os.path.splitext(args.source_file)[0] + ".jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in tqdm(data):
            example = dict(example)
            if 'label' not in example:
                example['label'] = 'CORRECT'
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")

def apply_transformation(data, operation):
    new_data = []
    for example in tqdm(data):
        try:
            new_example = operation.transform(example)
            if new_example:
                new_data.append(new_example)
        except Exception as e:
            print("Caught exception:", e)
    return new_data


def ko_ner(args, data):
    ner = Pororo(task="ner", lang="ko")
    for example in tqdm(data):
        ner_text = defaultdict(list)
        ner_summary = defaultdict(list)
        tmp = []
        lang_txt = ""
        stack = []
        for idx, txt in enumerate(example['text']):
            if (txt == "「" or txt == '《' or txt == '[' or txt == '(' or txt == "\'" or txt == '‘' or txt == '“') and not stack:
                stack.append(txt)
                if idx != 0 and example['text'][idx - 1] == ' ':
                    tmp += [(' ', 'O')]
                if not lang_txt:
                    lang_txt = txt
                    continue
                else:
                    tmp += ner(lang_txt)
                lang_txt = txt
            elif (txt == "「" or txt == '《' or txt == '[' or txt == '(' or txt == '‘' or txt == '“') and stack:
                stack.append(txt)
                lang_txt += txt

            elif (txt == '》' or txt == ']' or txt == "\'" or txt == ')' or txt == '’' or txt == '”') and stack:
                if txt == "\'" and "\'" in stack:
                    stack.remove(txt)
                elif txt == '」' and '「' in stack:
                    stack.remove('「')
                elif txt == '」' and '《' in stack:
                    stack.remove('《')
                elif txt == ')' and '(' in stack:
                    stack.remove('(')
                elif txt == ']' and '[' in stack:
                    stack.remove('[')
                elif txt == '’' and '‘' in stack:
                    stack.remove('‘')
                elif txt == '”' and '“' in stack:
                    stack.remove('“')
                lang_txt += txt
                if not stack:
                    tmp += ner(lang_txt)
                    if idx != len(example['text']) - 1 and example['text'][idx + 1] == ' ':
                        tmp += [(' ', 'O')]
                    lang_txt = ""
                else:
                    stack.append(txt)

            elif len(lang_txt) > 500 or idx == len(example['text']) - 1:
                lang_txt += txt
                tmp += ner(lang_txt)
                if txt == ' ':
                    tmp += [(' ', 'O')]
                if idx != len(example['text']) - 1 and example['text'][idx + 1] == ' ':
                    tmp += [(' ', 'O')]

                lang_txt = ""
            else:
                lang_txt += txt

        text_st = 0
        for tar, ent in tmp:
            if ent == 'O':
                text_st += len(tar)
                continue
            ner_text[ent].append((text_st, tar))
            text_st += len(tar)
        example["text_ner"] = ner_text

        summary_st = 0
        tmp = ner(example["summary"])
        for tar, ent in tmp:
            if ent == 'O':
                summary_st += len(tar)
                continue
            ner_summary[ent].append((summary_st, tar))
            summary_st += len(tar)
        example["summary_ner"] = ner_summary

    return data

class Backtranslation():
    def __init__(self, dst_lang=None):
        self.src_lang = "ko"
        self.dst_lang = dst_lang
        self.accepted_langs = ["fr", "de", "zh-TW", "es","en","ru"]
        self.translator = Translator()

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["summary"] is not None, "Claim must be available"

        new_example = dict(example)
        new_claim = self.__backtranslate(new_example["summary"])

        if new_claim:
            new_example["summary"] = new_claim
            new_example["backtranslation"] = True
            return new_example
        else:
            return None

    def __backtranslate(self, claim):
        # chose destination language, passed or random from list
        dst_lang = self.dst_lang if self.dst_lang else random.choice(self.accepted_langs)

        # translate to intermediate language and back
        claim_trans = self.translator.translate(claim, dest=dst_lang)
        claim_btrans = self.translator.translate(claim_trans.text, dest=self.src_lang)

        new_claim = claim_btrans.text
        if claim == new_claim:
            return None
        else:
            return new_claim

class NERSwap():
    def __init__(self):
        self.categories = ()

    def transform(self, example):
        assert example["text"] is not None, "Text must be available"
        assert example["summary"] is not None, "Summary must be available"

        new_example = dict(example)
        new_sum, aug_span = self.__swap_entities(example["text"], example["summary"], example["text_ner"], example["summary_ner"])

        if new_sum:
            new_example["corrupt_sum"] = new_sum
            new_example["label"] = LABEL_MAP[False]
            new_example["augmentation"] = self.__class__.__name__
            new_example["augmentation_span"] = aug_span
            return new_example
        else:
            return None

    def __swap_entities(self, text, summary, text_ner, summary_ner):
        if not text_ner or not summary_ner:
            return None, None
        summary_ents = [ent for ent in summary_ner.keys() if ent in self.categories]
        text_ents = [ent for ent in text_ner.keys() if ent in self.categories]
        tmp = random.choice(summary_ents) #ner 종류
        replace_ent = random.choice(summary_ner[tmp]) # 실제 이름
        candidate_ents = []

        for ent in text_ents:
            if ent == tmp:
                for ent_text in text_ner[ent]:
                    if replace_ent != ent_text:
                        candidate_ents.append(ent_text)

        if not candidate_ents:
            return None, None

        swapped_ent = random.choice(candidate_ents)
        st_point, ed_point = summary.find(replace_ent), summary.find(replace_ent)+len(replace_ent)
        summary_swp = summary[:st_point] + swapped_ent + summary[ed_point:]

        augmentation_span = (st_point, st_point+ len(swapped_ent)-1)

        return summary_swp, augmentation_span

class EntitySwap(NERSwap):
    def __init__(self):
        super().__init__()
        self.categories = ("PERSON","LOCATION","ORGANIZATION","ARTIFACT","CIVILIZATION", "ANIMAL", "PLANT", "STUDY_FIELD","THEORY", "EVENT", "MATERIAL")

class NumberSwap(NERSwap):
    def __init__(self):
        super().__init__()
        self.categories = ("DATE", "TIME","TERM","QUANTITY")

def main(args):
    data = load_json(args.source_file)
    # print(data)
    print("Ko_Named Entity Recognition")
    data = ko_ner(args, data)
    data_btrans = []
    # if not args.augmentations or "backtranslation" in args.augmentations:
    #     print("Creating backtranslation examples")
    #     btrans_op = Backtranslation()
    #     data_btrans = apply_transformation(data, btrans_op)
    #     print("Backtranslated %s example pairs." % len(data_btrans))
    #     save_data(args, data_btrans, "btrans")

    data_positive = data + data_btrans
    # save_data(args, data_positive, "positive")

    if not args.augmentations or "entity_swap" in args.augmentations:
        print("Creating entity swap examples")
        entswap_op = EntitySwap()
        data_entswp = apply_transformation(data, entswap_op)
        print("EntitySwap %s example pairs." % len(data_entswp))

        # save_data(args,data_entswp, "entswp")

    if not args.augmentations or "number_swap" in args.augmentations:
        print("Creating number swap examples")
        numswap_op = NumberSwap()
        data_numswp = apply_transformation(data, numswap_op)
        print("NumberSwap %s example pairs." % len(data_numswp))

        # save_data(args,data_numswp, "numswp")
    random.shuffle(data_entswp)
    random.shuffle(data_numswp)
    #
    # save_data(args, data_negative, "negative")
    #
    data_all = data_positive + data_numswp[:int(len(data_positive)/7*1.5)] + data_entswp[:int(len(data_positive)/7*1.5)]
    save_data(args, data_positive)
    print("Creating entity swap exemples")

if __name__ == "__main__":
        PARSER = argparse.ArgumentParser()
        PARSER.add_argument("--source_file", type=str, help="Path to file contains source documents.")
        PARSER.add_argument("--augmentations", type=str, nargs="+", default=(), help="List of data augmentation applied to data.")
        PARSER.add_argument("--all_augmentations", action="store_true", help="Flag whether all augmentation should be applied.")

        ARGS = PARSER.parse_args()

        main(ARGS)