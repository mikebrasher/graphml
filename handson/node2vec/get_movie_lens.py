#!/usr/bin/env python

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def main():
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    with urlopen(url) as stream:
        with ZipFile(BytesIO(stream.read())) as zipfile:
            zipfile.extractall('.')


if __name__ == '__main__':
    main()
