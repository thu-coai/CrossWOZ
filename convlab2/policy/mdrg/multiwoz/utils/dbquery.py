# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
import zipfile
from convlab2.util.file_util import cached_path


def auto_download():
    model_path = os.path.join(os.path.dirname(__file__), os.pardir,  'model')
    data_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
    db_path = os.path.join(os.path.dirname(__file__), os.pardir, 'db')
    root_path = os.path.join(os.path.dirname(__file__), os.pardir)

    urls = {model_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_model.zip',
            data_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_data.zip',
            db_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_db.zip'}

    for path in [model_path, data_path, db_path]:
        if not os.path.exists(path):
            file_url = urls[path]
            print('Downloading from: ', file_url)
            archive_file = cached_path(file_url)
            print('Extracting...')
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_path)


# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    auto_download()
    dbs[domain] = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'db/{}_db.json'.format(domain))))

def query(domain, constraints, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    if domain == 'taxi':
        return [{'taxi_colors': random.choice(dbs[domain]['taxi_colors']), 
        'taxi_types': random.choice(dbs[domain]['taxi_types']), 
        'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
    if domain == 'police':
        return dbs['police']
    if domain == 'hospital':
        return dbs['hospital']

    found = []
    for record in dbs[domain]:
        for key, val in constraints:
            if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                pass
            else:
                if key not in record:
                    continue
                if key == 'leaveAt':
                    val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                    val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                    if val1 > val2:
                        break
                elif key == 'arriveBy':
                    val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                    val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                    if val1 < val2:
                        break
                # elif ignore_open and key in ['destination', 'departure', 'name']:
                elif ignore_open and key in ['destination', 'departure']:
                    continue
                else:
                    if val.strip() != record[key].strip():
                        break
        else:
            found.append(record)

    return found



