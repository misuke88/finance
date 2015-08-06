from collections import Counter
import gzip
import re

from gensim import corpora, models, similarities
from nltk.corpus import stopwords

from settings import DATA_DIR, 8K_DIR
from utils import checkdir, file_read, get_today, get_version


KEYS = 'FILE TIME EVENTS TEXT ITEM'.split()


def get_id_docs_from_gz(company_code):

    def get_id_doc(doc):
        lines = filter(None, doc.split('\n'))
        id_ = lines[0].split('/')[-1].split('.')[0]
        doc = ' '.join([line for line in lines\
                if not (any(line.startswith(k) for k in KEYS) or line=='</DOCUMENT>')])
        return (id_, doc)


    with gzip.open('%s/%s.gz' % (8K_DIR, company_code)) as f:
        docs = filter(None, f.read().split("<DOCUMENT>"))

    return [get_id_doc(d) for d in docs]


def parse_doc(doc):
    # TODO: remove special characters
    doc = doc.lower()
    doc = doc.replace('\t', ' ').replace('\n', ' ')
    doc = re.sub('\s+', ' ', doc)
    return doc


def append_id_docs_to_file(id_docs, filename):
    with open(filename, 'a') as f:
        for i in id_docs:
            id_, doc = i[0], parse_doc(i[1])
            f.write('%s\t%s\n' % (id_, doc))


if __name__ == '__main__':

    company_code = 'C'
    filename = '%s/stock.tsv' % DATA_DIR

    open(filename, 'w').close()     # clear file
    id_docs = get_id_docs_from_gz(company_code)
    append_id_docs_to_file(id_docs, filename)
    print('%s\t%s' % (company_code, len(id_docs)))
