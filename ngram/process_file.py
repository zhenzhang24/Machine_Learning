import sys

f = open(sys.argv[1])
for line in f.readlines():
    print line.decode('gbk').encode('utf-8')
