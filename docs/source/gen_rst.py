import re
import os
import argparse

def process(text):
    result = []
    remove_module_contents = False
    for line in text:
        if re.match('Module contents', line) and remove_module_contents:
            break
        result.append(line)
        m = re.match(r'(\s+):show-inheritance:', line)
        if m:
            line = m.group(1) + ':special-members: __init__\n'
            result.append(line)
            remove_module_contents = True
    return result


def generate_rst(file_):
    with open(file_, 'r', encoding='utf-8') as f:
        text = f.readlines()

    text = process(text)
    with open(file_, 'w', encoding='utf-8') as f:
        for line in text:
            f.write(line)

def main(**kwargs):
    project = kwargs.get('project')
    if not project:
        return
    for file_ in os.listdir('.'):
        if os.path.isfile(file_) and file_.startswith(project) and file_.endswith('.rst'):
            generate_rst(file_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=str, help='project name')
    args = parser.parse_args()
    main(**vars(args))
