#!/bin/bash
#translation with quantized model
#st_tm=$(date +%s)
#python myscript/quant_test.py --model mymodel/tmp/model.zh-en_step_132000.pt -src /data/nlp/user/lsk/OpenNMT-py-master/mydata/valid_ds/val.zh -output $1 -replace_unk -verbose -qt_mdl_path /data/nlp/user/lsk/onmt_test/mymodel/tmp/mdl_only.qt.afti -report_time -train_config myconfig/tmp/config-transformer_big-zyb-zh-en-4GPU.yml -train_from mymodel/tmp/model.zh-en_step_132000.pt -mdl_tp $2
#end_tm=$(date +%s)
#echo "time total use : " $((end_tm - st_tm)) "s"

#python myscript/quant_test.py -train_config myconfig/tmp/train_cfg.yml -trans_config myconfig/tmp/trans_cfg.yml -output $1 -mdl_tp $2
python myscript/quant_test.py -config myconfig/tmp/quant_cfg.yml -output $1 -mdl_tp $2 -thd_num 1
