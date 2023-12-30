from argparse import ArgumentError
import os
import sys
import time
import json
import random
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from plot_grad import *
from pathlib import Path
import pathlib
from datetime import datetime
import tqdm
from dataset.dataset_iter import ContextGraphIterableDataset
from dynmodel.context_model import DynGraphContextModel
from dynmodel.context_model_types import DynGraphContextModelTypes

from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter 

torch.multiprocessing.set_sharing_strategy('file_system')
from utils import AverageMeter

from constants import *
import socket 

hostname = socket.gethostname()

#REMOVE_MODULE=True
# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


uddp=False
def main():
    args.dropout = 0.0
    if uddp:
        args.world_size = args.gpus
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8887'
        mp.spawn(run_inference, nprocs=args.gpus, args=(args,), join=True)
    else:
        run_inference(0, args)


def run_inference(gpu, args):
    rank = gpu
    if uddp:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
    # load data
    ldpath = Path(args.logdir)

    Path.mkdir(ldpath, parents=True, exist_ok=True)
    logfilename = os.path.join(args.logdir, f'{hostname}{gpu}.log')
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',level=logging.INFO, handlers=[logging.FileHandler(logfilename, 'w'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    ## save config
    curr_time = str(datetime.now())
    config=vars(args)
    logger.info(f'Saving config at {args.logdir}')
    json.dump(config, open(os.path.join(args.logdir, f'config_{curr_time}'), 'w'), indent=2)

    
    val_dataset = ContextGraphIterableDataset(data_path=args.data_path, split='test', is_test=False, gpu=gpu,total_gpu=args.gpus, logger=logger,iter_type='binary_batch')

    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=2,
                                       drop_last=False, collate_fn=val_dataset.collate_fn)
    print('Getting vocab...', gpu)
    gobal_syntax_vocabulary = val_dataset.gobal_syntax_vocabulary


    
    # load model
    if args.model_type== 'dyn':
        model = DynGraphContextModel(args, device=gpu, gobal_syntax_vocabulary=gobal_syntax_vocabulary)
        eval_method = run_infer
    elif args.model_type == 'dyn-type':
        model = DynGraphContextModelTypes(args, device=gpu, gobal_syntax_vocabulary=gobal_syntax_vocabulary)
        eval_method = run_infer_with_types
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if uddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    
    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint[EPOCH]
        best_val = checkpoint[BEST_VAL]
        temp_k = list(checkpoint[STATE_DICT].keys())[0]
        if 'module' in temp_k:
            print('Found module...')
            REMOVE_MODULE = True
        else:
            print('Not Found module...')
            REMOVE_MODULE = False
        if REMOVE_MODULE:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint[STATE_DICT].items():
                name = k[7:] # remove `module.`
                print(name)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint[STATE_DICT])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
    else:
        raise ArgumentError(f'resume is needed {args.resume}')


    logger.info('Loaders prepared.')
    logger.info(f"Unique tokens in logical form vocabulary: {len(gobal_syntax_vocabulary)}")
    logger.info(f'Batch: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')

    acc1, acc2 = eval_method(val_loader, model, gobal_syntax_vocabulary, logger)
    print(acc1, acc2)
    

def run_infer_with_types(val_loader, model, gobal_syntax_vocabulary, logger):
    # switch to evaluate mode
    model.eval()
    dl=args.val_dataset_size
    print(dl)
    recon_len = int(dl)
    print(recon_len)
    
    outfile = open(f'{args.path_results}/test_res.jsonl', 'w')
    string_acc_meter_beam = AverageMeter()
    string_acc_meter_beam_withoutavg = AverageMeter()
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(val_loader), total=recon_len):
            
            nodelinks = batch['nodelinks']
            logical_forms_str = batch['logical_forms_str']
            extra_links=batch['extra_and_type_links']
            s_input = batch['inputs']
            
            
            #output, target = model.predict_beam_search(batch, logger)
            output, padded_len_gnodes = model.predict_greedy(batch, logger)
            #output, target = model(batch, logger)
            string_acc,string_match_count, numofins = get_string_types(s_input, output, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=0, beam=True)
            string_acc_meter_beam.update(string_acc)
            string_acc_meter_beam_withoutavg.update(string_match_count, numofins)
            if recon_len < step:
                break

    return string_acc_meter_beam.avg, string_acc_meter_beam_withoutavg.avg


def get_string_types(s_input, output, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger, outfile, batch, is_print, beam=False):
    if beam:
        max_token_idx = output['predictions'][0]
    else:
        lfoutput = output
        max_token_idx = torch.argmax(lfoutput, dim = 2)
        print(len(max_token_idx))
    string_acc = 0
    kb_acc = 0
    for iter_idx in range(len(logical_forms_str)):
        foutput={}
        foutput['turnID'] = batch['turnIDs'][iter_idx]
        foutput['answer'] = batch['answers'][iter_idx]
        foutput['results'] = batch['results'][iter_idx]
        foutput['inputs'] = batch['inputs'][iter_idx]
        foutput['nodelinks'] = batch['nodelinks'][iter_idx]
        foutput['logical_forms_str'] = batch['logical_forms_str'][iter_idx]
        foutput['question-type'] = batch['question-types'][iter_idx]
        foutput['description'] = batch['description'][iter_idx]
        #foutput['node_lf_mappings'] = batch['node_lf_mappings'][iter_idx]

        current_token_idx = max_token_idx[iter_idx]
        out_str = []
        for t in current_token_idx:
            if t >= len(gobal_syntax_vocabulary):
                if t >= len(gobal_syntax_vocabulary) + padded_len_gnodes:
                    t = t - len(gobal_syntax_vocabulary) - padded_len_gnodes
                    s = extra_links[iter_idx][t]
                else:
                    t = t - len(gobal_syntax_vocabulary)
                    s = nodelinks[iter_idx][t]
            else:
                s = gobal_syntax_vocabulary.get_itos()[t]
            if s == EOS_TOKEN:
                break
            out_str.append(s)

        logical_form = []
        for action in logical_forms_str[iter_idx]:
            logical_form.append(action[1])

        #real_str = [t[1] if t[1] not in ENTITY, REL for t in logical_forms_str[iter_idx]]
        #real_kb_ele = [nodelinks[iter_idx][t] for t in node_lf_mappings[iter_idx]]
        foutput['predicted_lf'] = out_str
        #foutput['kb_elements'] = real_kb_ele

                
        out_str = ' '.join(out_str).lower()
        logical_form = ' '.join(logical_form).lower()
        if logical_form == out_str:
            string_acc += 1
            foutput['string_match'] = 1
        else:
            foutput['string_match'] = 0
    
        outfile.write(json.dumps(foutput) + '\n')
    
    
    string_match_count = string_acc
    string_acc = string_acc / len(logical_forms_str)
    logger.info(f'{beam} string_acc {string_acc}')
    return string_acc, string_match_count, len(logical_forms_str)


def run_infer(val_loader, model, gobal_syntax_vocabulary, logger):
    # switch to evaluate mode
    model.eval()
    dl=args.val_dataset_size
    recon_len = int(dl)
    
    outfile = open(f'{args.path_results}/test_res.jsonl', 'w')
    string_acc_meter_beam = AverageMeter()
    string_acc_meter_beam_withoutavg = AverageMeter()
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(val_loader), total=recon_len):
            
            node_lf_mappings = batch['node_lf_mappings']
            nodelinks = batch['nodelinks']
            logical_forms_str = batch['logical_forms_str']
            s_input = batch['inputs']
                        
            output = model.predict_greedy(batch, logger)

            string_acc,string_match_count, numofins = get_string(s_input, output, node_lf_mappings, nodelinks, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=0, beam=True)
            string_acc_meter_beam.update(string_acc)
            string_acc_meter_beam_withoutavg.update(string_match_count, numofins)
            if recon_len < step:
                break

    return string_acc_meter_beam.avg, string_acc_meter_beam_withoutavg.avg


def get_string(s_input, output, node_lf_mappings, nodelinks, gobal_syntax_vocabulary, logical_forms_str, logger, outfile, batch, is_print, beam=False):
    if beam:
        max_token_idx = output['predictions']
    else:
        lfoutput = output
        max_token_idx = torch.argmax(lfoutput, dim = 2)
    string_acc = 0
    kb_acc = 0
    for iter_idx in range(len(logical_forms_str)):
        foutput={} 
        foutput['fids'] = batch['fids'][iter_idx]
        foutput['inputs'] = batch['inputs'][iter_idx]
        foutput['answer'] = batch['answers'][iter_idx]
        foutput['results'] = batch['results'][iter_idx]        
        foutput['nodelinks'] = batch['nodelinks'][iter_idx]
        foutput['logical_forms_str'] = batch['logical_forms_str'][iter_idx]
        foutput['node_lf_mappings'] = batch['node_lf_mappings'][iter_idx]
        foutput['qtypes'] = batch['qtypes'][iter_idx]

        current_token_idx = max_token_idx[iter_idx]
        out_str = []
        for t in current_token_idx:
            if t >= len(gobal_syntax_vocabulary):
                t = t - len(gobal_syntax_vocabulary)
                s = nodelinks[iter_idx][t]
            else:
                s = gobal_syntax_vocabulary.get_itos()[t]
            if s == EOS_TOKEN:
                break
            out_str.append(s)

        logical_form = []
        for action in logical_forms_str[iter_idx]:
            logical_form.append(action[1])

        real_kb_ele = [nodelinks[iter_idx][t] for t in node_lf_mappings[iter_idx]]
        foutput['predicted_lf'] = out_str
        foutput['kb_elements'] = real_kb_ele

                
        out_str = ' '.join(out_str).lower()
        logical_form = ' '.join(logical_form).lower()
        if logical_form == out_str:
            string_acc += 1
            foutput['string_match'] = 1
        else:
            foutput['string_match'] = 0
    
        outfile.write(json.dumps(foutput) + '\n')
    
    
    string_match_count = string_acc
    string_acc = string_acc / len(logical_forms_str)
    return string_acc, string_match_count, len(logical_forms_str)


if __name__ == '__main__':
    pathlib.Path(args.snapshots).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_results).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_inference).mkdir(parents=True, exist_ok=True)
    main()
