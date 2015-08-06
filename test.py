import glob
import gzip
import re


from collections import Counter
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from pprint import pprint


from settings import TARGETPATH
from utils import checkdir, file_read, get_today, get_version

# def openfiles(TARGETPATH):

#     filenames = glob.glob('%s/*.gz' % TARGETPATH)
# 	for filename in filenames:
#         print filename
#         docs = file_read(filename)


if __name__ == '__main__':

	filenames = glob.glob('%s/*.gz' % TARGETPATH)
	documents = []
	for filename in filenames:
		path = '%s' %(filename)

		with gzip.open(path) as f:
			d = f.read().split("<DOCUMENT>")

		d.remove('') # remove null list

		for i in range(len(d)):
			item = d[i].replace('.', '').lower().split("item:")
			documents.append(item[-1])

	stop = set(stopwords.words('english'))
	texts = [[word for word in document.lower().split() if word not in stop] for document in documents]
	corpus = [dictionary.doc2bow(text) for text in texts]
	# pprint(texts)

	dictionary = corpora.Dictionary(texts)
	dictionary.save('/tmp/finance.dict')
	idList = dictionary.token2id


	print(dictionary)
	# items = []
	# # for i in range(len(d)):
	# item = d[3].replace('.', '').lower().split("item:")
	# term = re.findall(r'\w+',d[1])
	# c = Counter(term).most_common(10)

	# c = {(a, b) for a, b in term.items() if b>1}
	# print d[3], item[-1]
