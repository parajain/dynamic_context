import subprocess
import os, time, socket
from datetime import datetime
import tqdm, json
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from plot_grad import *
from pathlib import Path
import pathlib
from tracking import Tracker
from dataset.dataset_iter import ContextGraphIterableDataset

from dynmodel.context_model import DynGraphContextModel
from dynmodel.context_model_types import DynGraphContextModelTypes

from optimizer_facade import get_optimizer
from torch.utils.data import DataLoader
from constants import *
from torchtext.vocab import vocab
from collections import Counter
torch.multiprocessing.set_sharing_strategy('file_system')
from utils import (AverageMeter, save_checkpoint, init_weights)

hostname = socket.gethostname()

# set logger
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print('gpu mem ', gpu_memory_map)
    return gpu_memory_map

# set a seed value
use_ddp=args.gpus > 1
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main():
    if use_ddp:
        args.world_size = args.gpus
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8887'
        mp.spawn(trainer, nprocs=args.gpus, args=(args,), join=True)
    else:
        trainer(0, args)


def trainer(gpu, args):
    rank = gpu
    if use_ddp:
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
    def get_val_loader():
        val_dataset = ContextGraphIterableDataset(data_path=args.data_path, split='valid', is_test=False, gpu=gpu,total_gpu=args.gpus, logger=logger, iter_type=args.iter_type)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=val_dataset.collate_fn)
        return val_loader
    
    def get_train_loader():
        train_dataset = ContextGraphIterableDataset(data_path=args.data_path, split='train', is_test=False, gpu=gpu,total_gpu=args.gpus, logger=logger,iter_type=args.iter_type)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, drop_last=False, collate_fn=train_dataset.collate_fn)
        args.real_batch_size = train_dataset.real_batch_size
        return train_loader

    
    if gpu == 0:
        if args.schedule_loss_weight:
            loss_type='schedule_loss_weight'
        elif args.learn_loss_weights:
            loss_type='learn_loss_weights'
        else:
            loss_type = 'simple'
        comment = f'expname={args.expname} optim={args.optim} {loss_type} enc_layers={args.enc_layers} host={hostname}'
        tracker = Tracker(base_tb_dir=args.tbd,tensorboard_dir=f'{comment}', config=vars(args) ,log_filenames=['log'], key=comment, use_wandb=args.use_wandb, force_new=True)
        curr_time = str(datetime.now())
        config=vars(args)
        json.dump(config, open(os.path.join(args.snapshots, f'config_{curr_time}'), 'w'), indent=2)
        writer = tracker.writer
    else:
        writer = None

    
    print('Getting vocab... gpu: ', gpu)
    def load_target_vocab(tgt_vocab_file):
        gobalVocF = open(tgt_vocab_file)
        d = {}
        for l in gobalVocF.readlines():
            d[l.strip()] = 1
        gobal_syntax_vocabulary = vocab(Counter(d), specials=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, 'num', '0'])
        print("*\t Global syntax constants vocabulary size:", len(gobal_syntax_vocabulary))
        return gobal_syntax_vocabulary
    tgt_vocab_file = os.path.join(args.data_path, 'target_syntax.vocab')
    gobal_syntax_vocabulary = load_target_vocab(tgt_vocab_file)
    #writer = SummaryWriter(log_dir=args.tbd)
    train_loader = get_train_loader()
    val_loader = get_val_loader()


    
    # load model
    assert args.model_type == 'dyn-type'
    if args.model_type== 'dyn':
        model = DynGraphContextModel(args, device=gpu, gobal_syntax_vocabulary=gobal_syntax_vocabulary)
    elif args.model_type == 'dyn-type':
        model = DynGraphContextModelTypes(args, device=gpu, gobal_syntax_vocabulary=gobal_syntax_vocabulary)
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    # initialize model weights
    #init_weights(model)
    
    non_bert_params = []
    bert_parameters = []
    for n, p in model.named_parameters():
        print(n, p.requires_grad)
        if 'bert' in n:
            bert_parameters.append(p)
        else:
            non_bert_params.append(p)


    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # define loss function (criterion)
    #criterion = KGLoss(learn_loss_weights=args.learn_loss_weights , schedule_loss_weight=args.schedule_loss_weight,ignore_index=gobal_syntax_vocabulary.get_stoi()[PAD_TOKEN],  lambda1=args.lambda1, model_type=args.model_type)
    criterion = nn.CrossEntropyLoss(ignore_index=gobal_syntax_vocabulary.get_stoi()[PAD_TOKEN])
    
    if args.learn_loss_weights:
        logger.info('Learning loss interpolation weights...')

    # define optimizer
    dl=args.train_dataset_size
    total_batches = dl/args.real_batch_size

    lr=args.lr
    non_bert_param_group = {"params": non_bert_params, "lr": lr, "initial_lr": lr}
    param_groups = [non_bert_param_group]
    if args.fine_tune_bert:
        bert_lr=2e-5
        bert_param_group = {"params": bert_parameters, "lr": bert_lr, "weight_decay": 0}
        param_groups.append(bert_param_group)
    
    optimizer, scheduler = get_optimizer(args, model=model, decay_steps=total_batches*2)

    global_steps = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume, map_location='cuda:'+str(gpu))
            args.start_epoch = checkpoint[EPOCH]
            global_steps = checkpoint[GLOBAL_STEPS]
            best_val = checkpoint[BEST_VAL]
            model.load_state_dict(checkpoint[STATE_DICT])
            optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
            del checkpoint
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    logger.info('Loaders prepared.')
    logger.info(f"Unique tokens in logical form vocabulary: {len(gobal_syntax_vocabulary)}")
    logger.info(f'Batch: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')

    # run epochs
    #train_loader = get_train_loader()
    #val_loader = get_val_loader()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        global_steps = train(train_loader , val_loader, model, gobal_syntax_vocabulary, criterion, optimizer, scheduler, epoch, logger, writer, global_steps, gpu, args)       
        del train_loader
        del val_loader
        train_loader = get_train_loader()
        val_loader = get_val_loader() 
        # evaluate on validation set
        
        if (epoch+1) % args.valfreq == 0:
            logger.info('Validating Epoch ...')
            val_loader_epoch = get_val_loader()
            outfile = open(f'{args.path_results}/valid_res_{epoch}_{gpu}.jsonl', 'w')
            if args.model_type== 'dyn':
                val_loss, sacc = validate(val_loader_epoch, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val=300)
            else:
                val_loss, sacc = validate_types(val_loader_epoch, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val=300)
            
            if writer:
                writer.add_scalar(tag = 'val loss', scalar_value=val_loss, global_step=epoch)
                writer.add_scalar(tag = 'sacc', scalar_value=sacc, global_step=epoch)
            if val_loss < best_val:
                best_val = min(val_loss, best_val) # log every validation step

            opt_state = optimizer.state_dict()
            savef_filepath = save_checkpoint({
                EPOCH: epoch + 1,
                GLOBAL_STEPS: global_steps + 1,
                STATE_DICT: model.state_dict(),
                BEST_VAL: sacc,
                OPTIMIZER: opt_state,
                CURR_VAL: sacc})
            logger.info(f'* Val loss: {val_loss:.4f} String acc {sacc:.7f}')
            logger.info(f'Saved at {savef_filepath}')
            del val_loader_epoch
    #if writer:
    #    writer.close()

def train(train_loader, val_loader, model, gobal_syntax_vocabulary, criterion, optimizer,scheduler, epoch, logger, writer, global_steps, gpu, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    def reconstruct(gobal_syntax_vocabulary, lf_selected_indices, node_lf_mappings, target, links):
        out=[]
        c=0
        for iteridx, t in enumerate(target):
            if iteridx in lf_selected_indices:
                out.append(links[node_lf_mappings[c]])
                c+=1
            else:
                t=gobal_syntax_vocabulary.get_itos()[t]
                out.append(t)
        print(out)
    
    dl=args.train_dataset_size
    total_batches = dl/(args.real_batch_size * args.gpus)
    logger.info(f'Start training loop... {epoch} total_batches {total_batches} batch size {args.real_batch_size}')
    for step, batch in tqdm.tqdm(enumerate(train_loader), total=total_batches):
        # get inputs
        global_steps += 1

        '''
        if global_steps < 1900: continue

        num_nodes = max(d['input_ids'].shape[0] * d['input_ids'].shape[1] for d in batch['node_encodings'])
        if num_nodes > 10000:
            fids = batch['fids']
            get_gpu_memory_map()
            logger.info(f'Skipping {step} num nodes {num_nodes} {fids}')
            continue
        '''

        if global_steps % (5000 // args.gpus) == 0 and gpu == 0:
            gm=str(get_gpu_memory_map())
            logger.info(f'GPU mem map {gm}')
            outfile = open(f'{args.path_results}/valid_res_{epoch}_{gpu}_{global_steps}.jsonl', 'w')
            if args.model_type == 'dyn-type':
                val_loss, sacc = validate_types(val_loader, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val = 50)
            else:
                val_loss, sacc = validate(val_loader, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val = 50)
            model.train()

            opt_state = optimizer.state_dict()
            savef_filepath= save_checkpoint({
                    EPOCH: epoch + 1,
                    GLOBAL_STEPS: global_steps + 1,
                    STATE_DICT: model.state_dict(),
                    BEST_VAL: sacc,
                    OPTIMIZER: opt_state,
                    CURR_VAL: sacc})
            logger.info(f'Saved at {savef_filepath}')
            logger.info(f'* Val loss during training: {val_loss} String acc {sacc}')
            if writer:
                writer.add_scalar(tag = 'val loss train', scalar_value=val_loss, global_step=global_steps)
                writer.add_scalar(tag = 'sacc train', scalar_value=sacc, global_step=global_steps)
            
        
        fids = batch['turnIDs']
        
        try:
            if args.model_type == 'dyn-type':
                lfoutput, target, _ =  model(batch, logger)
            else:
                lfoutput, target =  model(batch, logger)
        except Exception as e:
            if 'out of memory' in str(e):
                logger.info(fids)
                logger.info(str(e))
                torch.cuda.empty_cache()
                model.zero_grad()
                raise e
            else:
                raise e
        
           
        lfoutput = lfoutput.contiguous().view(-1, lfoutput.shape[-1])
        #reconstruct(gobal_syntax_vocabulary, lf_selected_indices[0], node_lf_mappings[0], encoded_logical_form[0], nodelinks[0])
        target = target[:, 1:].contiguous().view(-1)
        device=lfoutput.device
        loss = criterion(lfoutput, target.to(device))
            
        losses.update(loss.data)
        optimizer.zero_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if global_steps % 100 == 0 and scheduler and global_steps > 10000:
            if args.optim == 'reduce':
                scheduler.step(metrics=loss.data)
            elif args.optim == 'cosine':
                scheduler.step()
            #else:
            #    scheduler.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        expected=(batch_time.avg*total_batches)/3600
        end = time.time()
           
        if writer and global_steps % 50 == 0:
            logger.info(f'Epoch: {epoch+1} -Batch No. {step+1} Train loss: {losses.val:.4f} ({losses.avg:.4f}) - Batch per: {((step+1)/total_batches)*100:.3f}% - ExpTime: {expected:0.2f}h')
            logger.info('Logical form Loss: {l1} '.format(l1=str(loss.data)))
            lrs = scheduler.get_last_lr()
            logger.info('lr {lr}'.format(lr=str(lrs)))
            writer.add_scalar("loss", loss.data, global_steps)
            if type(lrs) == list:
                for i, lr in enumerate(lrs):
                    writer.add_scalar("lr"+str(i),lr,  global_steps)
            else:
                writer.add_scalar("lr",lrs,  global_steps)

    return global_steps

def validate(val_loader, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val = 1000):
    losses = AverageMeter()
    string_acc_meter = AverageMeter()
    
    string_acc_meter_beam = AverageMeter()

    # switch to evaluate mode
    model.eval()
    dl=args.val_dataset_size
    recon_len = dl / 100
    
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step > max_val: break
            # get inputs
            #input_encodings, node_encodings, nodelinks, edges, encoded_logical_form, logical_forms_str, lf_selected_indices, node_lf_mappings, cum_num_nodes, fids = batch
            node_lf_mappings = batch['node_lf_mappings']
            nodelinks = batch['nodelinks']
            logical_forms_str = batch['logical_forms_str']
            s_input = batch['inputs']
            fids = batch['turnIDs']

            # compute output
            output, target = model(batch, logger)
            string_acc = get_string(s_input, output, node_lf_mappings, nodelinks, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=step % recon_len == 0)
            
            lfoutput = output
            lfoutput = lfoutput.contiguous().view(-1, lfoutput.shape[-1])
            #target = encoded_logical_form[:, 1:].contiguous().view(-1)
            target = target[:, 1:].contiguous().view(-1)
            device=lfoutput.device
            loss = criterion(lfoutput, target.to(device))
            
            string_acc_meter.update(string_acc)
            losses.update(loss.data)

            ### beam
            #output, target = model.predict_beam_search(batch, logger)
            #string_acc = get_string(s_input, output, node_lf_mappings, nodelinks, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=step % recon_len == 0, beam=True)
            
            string_acc_meter_beam.update(string_acc)

    outfile.close()
    #logger.info(f'* Val loss during training: {losses.avg} String acc {string_acc_meter.avg}')
    #logger.info(f'* Beam String acc {string_acc_meter_beam.avg}')
    return losses.avg, string_acc_meter.avg#, losses_beam.avg, string_acc_meter_beam.avg

def get_string(s_input, output, node_lf_mappings, nodelinks, gobal_syntax_vocabulary, logical_forms_str, logger, outfile, batch, is_print, beam=False):
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
        foutput['turnIDs'] = batch['turnIDs'][iter_idx]
        foutput['inputs'] = batch['inputs'][iter_idx]
        foutput['nodelinks'] = batch['nodelinks'][iter_idx]
        foutput['logical_forms_str'] = batch['logical_forms_str'][iter_idx]
        foutput['node_lf_mappings'] = batch['node_lf_mappings'][iter_idx]

        current_token_idx = max_token_idx[iter_idx]
        #out_str = [gobal_syntax_vocabulary.get_itos()[t] for t in current_token_idx]
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

        #real_str = [t[1] if t[1] not in ENTITY, REL for t in logical_forms_str[iter_idx]]
        real_kb_ele = [nodelinks[iter_idx][t] for t in node_lf_mappings[iter_idx]]
        foutput['predicted_lf'] = out_str
        foutput['kb_elements'] = real_kb_ele
        if is_print:
            logger.info(f' {beam} Input {s_input[iter_idx]}')
            logger.info(f' {beam} Predicted string {out_str}')
            logger.info(f' {beam} Real {logical_form}')
                
        out_str = ' '.join(out_str).lower()
        logical_form = ' '.join(logical_form).lower()
        if logical_form == out_str:
            logger.info(' {beam} local str match True')
            string_acc += 1
            foutput['string_match'] = 1
        else:
            foutput['string_match'] = 0
    
        outfile.write(json.dumps(foutput) + '\n')
    
    
    string_acc = string_acc / len(logical_forms_str)
    logger.info(f'{beam} string_acc {string_acc}')
    return string_acc



def validate_types(val_loader, model, gobal_syntax_vocabulary, criterion, outfile, logger, epoch, max_val = 500):
    losses = AverageMeter()
    string_acc_meter = AverageMeter()
    
    string_acc_meter_beam = AverageMeter()

    # switch to evaluate mode
    model.eval()
    dl=args.val_dataset_size
    recon_len = dl / 100
    
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step > max_val: break
            # get inputs
            #input_encodings, node_encodings, nodelinks, edges, encoded_logical_form, logical_forms_str, lf_selected_indices, node_lf_mappings, cum_num_nodes, fids = batch
            #encoded_logical_form = batch['encoded_logical_form']
            #node_lf_mappings = batch['node_lf_mappings']
            nodelinks = batch['nodelinks']
            extra_links=batch['extra_and_type_links']
            logical_forms_str = batch['logical_forms_str']
            s_input = batch['inputs']
            fids = batch['turnIDs']

            # compute output
            output, target, padded_len_gnodes = model(batch, logger)
            tempp={}
            tempp['predictions']=[target]
            #if epoch < 1:
            #    _ = check_types(s_input, tempp, None, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=0, beam=True)
            string_acc = get_string_types(s_input, output, None, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger,outfile, batch, is_print=0)
            
            lfoutput = output
            lfoutput = lfoutput.contiguous().view(-1, lfoutput.shape[-1])
            #target = encoded_logical_form[:, 1:].contiguous().view(-1)
            target = target[:, 1:].contiguous().view(-1)
            device=lfoutput.device
            loss = criterion(lfoutput, target.to(device))
            
            string_acc_meter.update(string_acc)
            losses.update(loss.data)

            string_acc_meter_beam.update(string_acc)

    outfile.close()
    return losses.avg, string_acc_meter.avg#, losses_beam.avg, string_acc_meter_beam.avg

def check_types(s_input, output, node_lf_mappings, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger, outfile, batch, is_print, beam=False):
    if beam:
        max_token_idx = output['predictions'][0]
    else:
        lfoutput = output
        max_token_idx = torch.argmax(lfoutput, dim = 2)
    string_acc = 0
    
    for iter_idx in range(len(logical_forms_str)):
        current_token_idx = max_token_idx[iter_idx][1:]
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
                
        out_str = ' '.join(out_str).lower()
        logical_form = ' '.join(logical_form).lower()
        if logical_form == out_str or '[unk]' in out_str or 'missingnode' in out_str:
            string_acc += 1
        else:
            logger.info('Does not match..........')
            logger.info(f'{logical_form} does not match')
            logger.info(f'{out_str}')
            import sys
            sys.exit(0)
        
    string_acc = string_acc / len(logical_forms_str)
    return string_acc


def get_string_types(s_input, output, node_lf_mappings, nodelinks, extra_links, padded_len_gnodes, gobal_syntax_vocabulary, logical_forms_str, logger, outfile, batch, is_print, beam=False):
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
        foutput['turnIDs'] = batch['turnIDs'][iter_idx]
        foutput['inputs'] = batch['inputs'][iter_idx]
        foutput['nodelinks'] = batch['nodelinks'][iter_idx]
        foutput['logical_forms_str'] = batch['logical_forms_str'][iter_idx]
        #foutput['node_lf_mappings'] = batch['node_lf_mappings'][iter_idx]

        current_token_idx = max_token_idx[iter_idx]
        #out_str = [gobal_syntax_vocabulary.get_itos()[t] for t in current_token_idx]
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
        if is_print:
            logger.info(f' {beam} Input {s_input[iter_idx]}')
            logger.info(f' {beam} Predicted string {out_str}')
            logger.info(f' {beam} Real {logical_form}')
                
        out_str = ' '.join(out_str).lower()
        logical_form = ' '.join(logical_form).lower()
        if logical_form == out_str:
            #logger.info(' {beam} local str match True')
            string_acc += 1
            foutput['string_match'] = 1
        else:
            foutput['string_match'] = 0
    
        outfile.write(json.dumps(foutput) + '\n')
    
    
    string_acc = string_acc / len(logical_forms_str)
    #logger.info(f'{beam} string_acc {string_acc}')
    return string_acc


if __name__ == '__main__':
    pathlib.Path(args.snapshots).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_results).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_inference).mkdir(parents=True, exist_ok=True)
    main()
