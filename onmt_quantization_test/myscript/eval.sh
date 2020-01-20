#!/bin/bash
#perl tools/multi-bleu.perl /home/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/ref1 < $1
perl tools/multi-bleu.perl -lc /home/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/ref1 /home/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/ref2 /home/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/ref3 /home/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/ref4 < $1
