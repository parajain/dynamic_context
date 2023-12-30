import json
from tqdm import tqdm
import time
import os, logging
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.utils.data import IterableDataset
import numpy as np
import itertools
from constants import *
#try:
#    from torchtext.data import Field, Example, Dataset
#except:
#   from torchtext.legacy.data import Field, Example, Dataset
from torchtext.vocab import vocab
from collections import Counter

CHUNKED_ITER_JSONL = 'chunked_jsonl'
ITER_JSONL= 'jsonl'
ITER_BIN= 'binary_batch'


class ChunkedFileIterator:
    def __init__(self, filename,n = 100000):
        self.filename = filename
        self.line_iter = None
        self.n = n

    def __iter__(self):
        return self
    
    def __next__(self):
        n=self.n
        if self.line_iter is None:
            self.fptr = open(self.filename)
            #nextlines = list(itertools.islice(self.fptr, n)) # can remove iter and list as islice will return an iter but we want to force read from file and keep it in buffer
            nextlines = itertools.islice(self.fptr, n)
            #print('FIrst Iteration', nextlines[:10])
            self.line_iter = iter(nextlines)
        try:
            line = next(self.line_iter)
            return line
        except StopIteration as e:
            nextlines = list(itertools.islice(self.fptr, n))
            #print('StopIteration', nextlines[:10])
            self.line_iter = iter(nextlines)
            line = next(self.line_iter)
            return line

class ShardBinaryBatchIterator:
    def __init__(self, file_list, logger):
        self.file_list_iter = iter(file_list)
        self.line_iter = None
        self.logger = logger
        '''logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%d/%m/%Y %I:%M:%S %p',
            level=logging.INFO,
            handlers=[
                #logging.FileHandler(f'{args.path_results}/train_{args.task}.log', 'w'),
                logging.FileHandler('filereadlog.log', 'w'),
                logging.StreamHandler()
            ])
        self.logger = logging.getLogger(__name__)'''

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.line_iter is None:
            next_file = next(self.file_list_iter)
            #bname = os.path.basename(next_file)
            #with open(bname, 'a') as bnf:
            #   bnf.write(next_file + '\n')
            #self.logger.info(f'\nLoading from initial {next_file}\n')
            self.nextlines = torch.load(next_file)
            #self.logger.info('New lines {l}'.format(l=len(self.nextlines)))
            self.line_iter = iter(self.nextlines)
        try:
            #self.logger.info('trying')
            line = next(self.line_iter)
            return line
        except StopIteration as e:
            next_file = next(self.file_list_iter)
            #self.logger.info(f'\nLoading from {next_file}\n')
            #bname = os.path.basename(next_file)
            #with open(bname, 'a') as bnf:
            #    bnf.write(next_file + '\n')
            #time.sleep(1)
            #del self.nextlines
            self.nextlines = torch.load(next_file)
            #self.logger.info('New lines {l}'.format(l=len(self.nextlines)))
            self.line_iter = iter(self.nextlines)
            line = next(self.line_iter)
            return line
    
    def __nextgen__(self):
        # This cannot be used as generator cannot be pickled
        for filename in self.file_list:
            self.logger.info(f'Loading {filename}')
            local_file=torch.load(filename)
            self.logger.info(f'Local data len {len(local_file)}')
            for sample in enumerate(local_file):
                self.logger.info(f'before yield {sample[0]}')
                yield sample



class ContextGraphIterableDataset(IterableDataset):
    def __init__(self, data_path, split, is_test, gpu, total_gpu, logger, iter_type=ITER_BIN):
        #assert iter_type in [CHUNKED_ITER_JSONL, ITER_BIN, ITER_JSONL] ## cleanup other iterators
        assert iter_type in [ITER_BIN]
    
        self.DEVICE='cuda:0'
        self.gpu=gpu
        self.logger = logger
        self.total_gpu = total_gpu
        self.data_path = data_path
        self.is_test = is_test
        self.split = split
        
        #self.filename =  'testfile.txt'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tgt_vocab_file = os.path.join(self.data_path, 'target_syntax.vocab')
        self.load_target_vocab(tgt_vocab_file)
        if iter_type == CHUNKED_ITER_JSONL:
            logger.info('Chunked...')
            self.filename =  os.path.join(self.data_path, self.split, 'all'+split+'.jsonl')
            self.file_iter = ChunkedFileIterator(self.filename, n = 100000)
            self.collate_fn = self.collate_fn_raw_jsonl
            self.preprocess = self.preprocess_jsonl
        elif iter_type == ITER_JSONL:
            self.filename =  os.path.join(self.data_path, self.split, 'all'+split+'.jsonl')
            print('Loading file ', self.filename)
            self.file_iter = None
            self.collate_fn = self.collate_fn_raw_jsonl
            self.preprocess = self.preprocess_jsonl
        else:
            self.file_list =  glob(os.path.join(self.data_path, self.split) + '/' + self.split + '*.pt')
            logger.info('Total number of bin files for {s} are {l}'.format(s=self.split, l=len(self.file_list)))

            data = torch.load(self.file_list[0])
            self.real_batch_size = data[0]['input_encodings']['input_ids'].size(0)
            logger.info(f'Real batch size {self.real_batch_size}')
            if len(self.file_list) < 1:
                logger.info('Something wrong with the data path no files were loaded')
            self.file_iter = ShardBinaryBatchIterator(self.file_list, logger=logger)
            self.collate_fn = self.collate_fn_binary_batch
            self.preprocess = self.preprocess_bin_batch
            
            



    def load_target_vocab(self, tgt_vocab_file):
        gobalVocF = open(tgt_vocab_file)
        d = {}
        for l in gobalVocF.readlines():
            d[l.strip()] = 1
        self.gobal_syntax_vocabulary = vocab(Counter(d), specials=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, 'num', '0'])
        print("*\t Global syntax constants vocabulary size:", len(self.gobal_syntax_vocabulary))
    
        
    def preprocess_jsonl(self, text):
        text_pp = text.rstrip()
        text_pp = json.loads(text_pp)
        return text_pp

    def preprocess_bin_batch(self, line):
        return line

    def line_mapper(self, line):
        #return line  
        text = self.preprocess(line)
        return text


        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # there is only one thread..
            #assert self.gpu == 0
            self.logger.info(f'Single data iter')
            worker_total_num=self.total_gpu
            mapped_itr = map(self.line_mapper, self.file_iter)
            #mapped_itr = enumerate(mapped_itr)
            global_worker_id =   self.gpu
            self.logger.info(f'Will start at {global_worker_id}, stop:None, step : {worker_total_num}')
            if worker_total_num > 1:
                mapped_itr = itertools.islice(enumerate(mapped_itr), global_worker_id, None, worker_total_num) # this will return line number too
            else:
                mapped_itr = enumerate(mapped_itr) # this will return line number too
        else:
            worker_total_num = worker_info.num_workers * self.total_gpu 
            global_worker_id = worker_info.id + self.gpu * worker_info.num_workers
            #self.logger.info('global_worker_id ' + str(global_worker_id) +' -  '+ str(self.gpu))
            if self.file_iter is None:
                self.file_iter = open(self.filename)
            mapped_itr = map(self.line_mapper, self.file_iter)            
            #Add multiworker functionality, this will return  index
            self.logger.info(f'{global_worker_id}, None, {worker_total_num}')
            mapped_itr = itertools.islice(enumerate(mapped_itr), global_worker_id, None, worker_total_num) # this will return line number too
            #mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)
        return mapped_itr
        
   
    def collate_fn_binary_batch(self, inp_batch):
        assert len(inp_batch) == 1, 'This is used when we have batch already created, but for dataloader bs should be set to 1. Real batch size comes from len in binary'
        iteridx = inp_batch[0][0]
        inp_batch = inp_batch[0][1]
        return inp_batch

    def pad_and_encode_logical_form(self, logical_form_token_list, nodelinks):
        encoded_logical_form_list = []
        lf_selected_indices, node_lf_mappings = [], []
        assert len(nodelinks) == len(logical_form_token_list)
        for lftokens, links in zip(logical_form_token_list, nodelinks):
            lf_sel = []
            nlf_map = []
            lf_vocab_encodings= [self.gobal_syntax_vocabulary.get_stoi()[BOS_TOKEN]]
            
            #lf_vocab_encodings.extend([self.gobal_syntax_vocabulary.get_stoi()[l] if l in self.gobal_syntax_vocabulary.get_stoi().keys() else self.gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN] for l in lftokens ])
            for iteridx, (toktype, tok) in enumerate(lftokens):
                if toktype in [ENTITY, RELATION, TYPE]:
                    lf_sel.append(iteridx+1) # +1 for BOS_TOKEN, BOS_TOKEN is passed into the transformer model
                    if tok in links:
                        nlf_map.append(links.index(tok))
                    else:
                        nlf_map.append(links.index('MISSINGNODE'))
                    lf_vocab_encodings.append(self.gobal_syntax_vocabulary.get_stoi()[toktype])
                # if token type is an action we will put real action index from vocab
                elif tok in self.gobal_syntax_vocabulary.get_stoi().keys():
                    lf_vocab_encodings.append(self.gobal_syntax_vocabulary.get_stoi()[tok])
                else:
                    lf_vocab_encodings.append(self.gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN]) # why would this happen ?
                

            lf_vocab_encodings.append(self.gobal_syntax_vocabulary.get_stoi()[EOS_TOKEN])
            lf_vocab_encodings = torch.tensor(lf_vocab_encodings, dtype=torch.int64)#, device=self.DEVICE)
            #lf_vocab_encodings = torch.cuda.LongTensor(lf_vocab_encodings)#, device=self.DEVICE)
            encoded_logical_form_list.append(lf_vocab_encodings)
            lf_selected_indices.append(lf_sel)
            node_lf_mappings.append(nlf_map)
            
        encoded_tensor = pad_sequence(encoded_logical_form_list, batch_first=True, padding_value=self.gobal_syntax_vocabulary.get_stoi()[PAD_TOKEN])
        return encoded_tensor, lf_selected_indices, node_lf_mappings

    
def test(data_path):
    from torch.utils.data import DataLoader
    dataset = ContextGraphIterableDataset(data_path=data_path, split='valid', is_test=False)
    dataloader = DataLoader(dataset=dataset, batch_size=3, num_workers=2,
                                       drop_last=False, collate_fn=dataset.collate_fn)
    for d in dataloader:
        print(d)

