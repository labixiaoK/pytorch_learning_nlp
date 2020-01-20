#!/bin/bash
#st_tm=$(date +%s)
python translate.py -model $1 -src  /data/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/val.zh -output $2 -replace_unk -verbose -report_time
#end_tm=$(date +%s)
#echo "time total use : " $((end_tm - st_tm)) "s"
