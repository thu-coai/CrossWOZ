#!/usr/bin/env python
# coding: utf-8
"""This script is used for removing some elements in py-modindex.html."""

import os
from bs4 import BeautifulSoup


def main(**kwargs):
    filepath = kwargs.get("file") or os.path.join(kwargs.get("doc_dir", ""), "py-modindex.html")
    if not os.path.isfile(filepath):
        raise RuntimeError('`{}` not exists'.format(filepath))
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        body = soup.find("div", attrs=dict(itemprop="articleBody"))
        assert body.div['class'][0] == "modindex-jumpbox"
        body.div.decompose()
        assert body.table.tr['class'][0] == 'pcap'
        body.table.tr.decompose()
        assert body.table.tr['class'][0] == 'cap'
        body.table.tr.decompose()
    except BaseException:
        raise RuntimeError
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(soup))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--file', '-f', type=str, default=None,
                        help="the html file that will be modified. If None, `doc_dir` must be provided.")
    parser.add_argument('--doc_dir', '-d', type=str, default=None, help="the dir that contains docs.")
    args = parser.parse_args()
    main(**vars(args))
