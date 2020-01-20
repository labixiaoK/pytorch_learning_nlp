#!/bin/bash
python preprocess.py -config myconfig/config-preps-zh-en.yml

#python preprocess.py \
#    -train_src ./mydata/train.zn.txt.sf \
#    -train_tgt ./mydata/train.en.txt.sf \
#    -valid_src ./mydata/valid_ds/val.zh \
#    -valid_tgt ./mydata/valid_ds/ref1 \
#    -save_data ./mydata/model_data/zh-en.data \
#	-src_seq_length 65 \
#	-tgt_seq_length 65 \
#   -src_vocab_size 10000 \
#    -tgt_vocab_size 50000
