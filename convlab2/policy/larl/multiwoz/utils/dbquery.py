# Modified by Microsoft Corporation.
# Licensed under the MIT license.

"""
"""
import json
import os
import random

# loading databases
domains = ['restaurant', 'hotel', 'attraction',
           'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))),
        'data', 'multiwoz', 'db', '{}_db.json'.format(domain))
    dbs[domain] = json.load(open(db_path))

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
                    val1 = int(val.split(':')[0]) * \
                        100 + int(val.split(':')[1])
                    val2 = int(record['leaveAt'].split(':')[0]) * \
                        100 + int(record['leaveAt'].split(':')[1])
                    if val1 > val2:
                        break
                elif key == 'arriveBy':
                    val1 = int(val.split(':')[0]) * \
                        100 + int(val.split(':')[1])
                    val2 = int(record['arriveBy'].split(':')[0]) * \
                        100 + int(record['arriveBy'].split(':')[1])
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
