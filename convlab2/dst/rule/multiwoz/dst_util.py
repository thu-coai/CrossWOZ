import re
from difflib import SequenceMatcher


def str_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _log(info):
    with open('fuzzy_recognition.log', 'a+') as f:
        f.write('{}\n'.format(info))
    f.close()


def minDistance(word1, word2):
    """The minimum edit distance between word 1 and 2."""
    if not word1:
        return len(word2 or '') or 0
    if not word2:
        return len(word1 or '') or 0
    size1 = len(word1)
    size2 = len(word2)
    tmp = list(range(size2 + 1))
    value = None
    for i in range(size1):
        tmp[0] = i + 1
        last = i
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
            last = tmp[j + 1]
            tmp[j + 1] = value
    return value


def normalize_value(value_set, domain, slot, value):
    """Normalized the value produced by NLU module to map it to the ontology value space.

    Args:
        value_set (dict):
            The value set of task ontology.
        domain (str):
            The domain of the slot-value pairs.
        slot (str):
            The slot of the value.
        value (str):
            The raw value detected by NLU module.
    Returns:
        value (str): The normalized value, which fits with the domain ontology.
    """
    slot = slot.lower()
    value = value.lower()
    value = ' '.join(value.split())
    try:
        assert domain in value_set
    except:
        raise Exception('domain <{}> not found in value set'.format(domain))
    if slot not in value_set[domain]:
        return value
        # raise Exception(
        #     'slot <{}> not found in db_values[{}]'.format(
        #         slot, domain))
    value_list = value_set[domain][slot]
    # exact match or containing match
    v = _match_or_contain(value, value_list)
    if v is not None:
        return v
    # some transfomations
    cand_values = _transform_value(value)
    for cv in cand_values:
        v = _match_or_contain(cv, value_list)
        if v is not None:
            return v
    # special value matching
    v = special_match(domain, slot, value)
    if v is not None:
        return v
    _log(
        'Failed: domain {} slot {} value {}, raw value returned.'.format(
            domain,
            slot,
            value))
    return value


def _transform_value(value):
    cand_list = []
    # a 's -> a's
    if " 's" in value:
        cand_list.append(value.replace(" 's", "'s"))
    # a - b -> a-b
    if " - " in value:
        cand_list.append(value.replace(" - ", "-"))
    # center <-> centre
    if value == 'center':
        cand_list.append('centre')
    elif value == 'centre':
        cand_list.append('center')
    # the + value
    if not value.startswith('the '):
        cand_list.append('the ' + value)
    return cand_list


def _match_or_contain(value, value_list):
    """match value by exact match or containing"""
    if value in value_list:
        return value
    for v in value_list:
        if v in value or value in v:
            return v
    # fuzzy match, when len(value) is large and distance(v1, v2) is small
    for v in value_list:
        d = minDistance(value, v)
        if (d <= 2 and len(value) >= 10) or (d <= 3 and len(value) >= 15):
            return v
    return None


def special_match(domain, slot, value):
    """special slot fuzzy matching"""
    matched_result = None
    if slot == 'arriveby' or slot == 'leaveat':
        matched_result = _match_time(value)
    elif slot == 'price' or slot == 'entrance fee':
        matched_result = _match_pound_price(value)
    elif slot == 'trainid':
        matched_result = _match_trainid(value)
    elif slot == 'duration':
        matched_result = _match_duration(value)
    return matched_result


def _match_time(value):
    """Return the time (leaveby, arriveat) in value, None if no time in value."""
    mat = re.search(r"(\d{1,2}:\d{1,2})", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


def _match_trainid(value):
    """Return the trainID in value, None if no trainID."""
    mat = re.search(r"TR(\d{4})", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


def _match_pound_price(value):
    """Return the price with pounds in value, None if no trainID."""
    mat = re.search(r"(\d{1,2},\d{1,2} pounds)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    mat = re.search(r"(\d{1,2} pounds)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    if "1 pound" in value.lower():
        return '1 pound'
    if 'free' in value:
        return 'free'
    return None


def _match_duration(value):
    """Return the durations (by minute) in value, None if no trainID."""
    mat = re.search(r"(\d{1,2} minutes)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


if __name__ == "__main__":
    # value_set = json.load(open('../../../data/multiwoz/db/db_values.json'))
    # print(normalize_value(value_set, 'restaurant', 'address', 'regent street city center'))
    print(
        minDistance(
            "museum of archaeology and anthropology",
            "museum of archaelogy and anthropology"))
