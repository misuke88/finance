import gzip


with gzip.open("D:/dropbox/Dropbox/DMLAB/2015_dada/finance/8K.tar/8K-gz/A.gz") as f:
	d = f.read().split("<DOCUMENT>")
print 'hello'
print d[1]