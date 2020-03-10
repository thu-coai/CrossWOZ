import json
import zipfile
import os
from convlab2.util.crosswoz.lexicalize import delexicalize_da


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def gen_da_voc(data):
    usr_da_voc, sys_da_voc = {}, {}
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages']):
            if turn['role'] == 'usr':
                da_voc = usr_da_voc
            else:
                da_voc = sys_da_voc
            for da in delexicalize_da(turn['dialog_act']):
                da_voc[da] = 0
    return sorted(usr_da_voc.keys()), sorted(sys_da_voc.keys())


if __name__ == '__main__':
    data = read_zipped_json('train.json.zip','train.json')
    usr_da_voc, sys_da_voc = gen_da_voc(data)
    json.dump(usr_da_voc, open('usr_da_voc.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(sys_da_voc, open('sys_da_voc.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

