import sys
import os
import time
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

import onmt
#import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_model
from onmt.utils.misc import set_random_seed
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import opts

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def load_model(opt, device_id):
    configure_process(opt, device_id)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    print('Loading checkpoint from %s' % opt.train_from)
    checkpoint = torch.load(opt.train_from,
                            map_location=lambda storage, loc: storage)
    print('======lsk======ckpt.keys(): {}'.format(str(checkpoint.keys())))

    for key in checkpoint.keys():
        print('====lsk====print_size_of_key: {}'.format(key))
        #json.dump(checkpoint[key], "temp.p")
        torch.save(checkpoint[key], "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')


    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    print('Loading vocab from checkpoint at %s.' % opt.train_from)
    vocab = checkpoint['vocab']

    fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                print(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    print('encoder: %d' % enc)
    print('decoder: %d' % dec)
    print('* number of parameters: %d' % n_params)
    print('====lsk====model: {}'.format(model))
    _check_save_model_path(opt)

    return model, checkpoint


def print_size_of_model(model):
    print('====lsk====print_size_of_model')
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def qt_and_save():
    #parser, _ = _get_parser('train')
    parser = _get_parser()

    opt = parser.parse_args()

    device_id = -1

    #load model
    model, checkpoint = load_model(opt, device_id)

    #quantize
    #quantized_model = torch.quantization.quantize_dynamic(model, {nn.Embedding, nn.Linear}, dtype=torch.qint8)
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Embedding, nn.Linear}, dtype=torch.qint8)
    print('====lsk====qted_model: {}'.format(quantized_model))
    print_size_of_model(model)
    print_size_of_model(quantized_model)

    #save only model, not checkpoint
    torch.save(quantized_model.state_dict(), opt.save_model)
    #not support torch.save(entire model) for now
    #torch.save(quantized_model, opt.save_model)
    print('====lsk====save qted_model only to : {}'.format(opt.save_model))

    return model, quantized_model


def translate(opt, mdl):
    ArgumentParser.validate_translate_opts(opt)
    #logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)

    ##!!!!set the quantized_model or original model
    translator.model = mdl

    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    
    torch.set_num_threads(opt.thd_num)
    print('====lsk====torch parall setting : {}'.format(torch.__config__.parallel_info()))

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        print("Translating shard %d." % i)
        sys.stdout.flush()
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )

def trans():
    parser = _get_parser()

    opt = parser.parse_args()

    model, _ = load_model(opt, -1)

    if opt.mdl_tp == 'ori':
        mdl = model
    else:
        print('====lsk====need to replace the original model with the quantized model in the translate()')
        quantized_model = torch.quantization.quantize_dynamic(model, {nn.Embedding, nn.Linear}, dtype=torch.qint8)
        mdl = quantized_model

    mdl.eval()
    print('====lsk====trans_model: {}'.format(mdl))
    sys.stdout.flush()
    translate(opt, mdl)



def config_opts(parser):
    parser.add('-trans_config', '--trans_config', required=False,
               is_config_file_arg=True, help='translate config file path')
    parser.add('-train_config', '--train_config', required=False,
               is_config_file_arg=True, help='train config file path')
    #parser.add('-save_config', '--save_config', required=False,
    #          is_write_out_config_file_arg=True,
    #           help='config file save path')

def _get_parser():
    parser = ArgumentParser(description='quantization.py')
    
    opts.config_opts(parser)
    opts.quant_opts(parser)

    '''
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opts.translate_opts(parser)
    '''
    #trans_parser.add('-qt_mdl_path', '--qt_mdl_path', type=str, default='/data/nlp/user/lsk/onmt_test/mymodel/tmp/mdl_only.qt.aft', help='quantized model load path')
    parser.add('-mdl_tp', '--mdl_tp', type=str, default='qt', help='quantized model or original model')
    parser.add('-thd_num', '--thd_num', type=int, default=40, help='torch thread num')

    return parser

def main():
    #qt_and_save()
    trans()


if __name__ == '__main__':
    main()







