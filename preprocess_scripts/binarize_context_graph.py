from concurrent.futures import process
import sys, os
from glob import glob
from multiprocessing import Pool
import json
#import matplotlib.pyplot as plt
import numpy as np
import traceback
from collections import OrderedDict
import pickle
import argparse
import gc
from tqdm import tqdm
import itertools
import torch
from torch.nn.utils.rnn import pad_sequence
torch.multiprocessing.set_sharing_strategy('file_system')

from transformers import BertTokenizer
from torchtext.vocab import vocab
from collections import Counter
# action related
ENTITY = 'entity'
RELATION = 'relation'
TYPE = 'type'
VALUE = 'value'
PREV_ANSWER = 'prev_answer'
ACTION = 'action'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
NA_TOKEN = 'NA'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

TRAIN=False
SKIP_MISSING=False
USE_POOL=False
ANS_CONTEXT=True

def pad_and_encode_logical_form(logical_form_token_list, nodelinks, tokenizer, gobal_syntax_vocabulary):
    encoded_logical_form_list = []
    lf_selected_indices, node_lf_mappings = [], []
    assert len(nodelinks) == len(logical_form_token_list)
    for lftokens, links in zip(logical_form_token_list, nodelinks):
        lf_sel = []
        nlf_map = []
        lf_vocab_encodings= [gobal_syntax_vocabulary.get_stoi()[BOS_TOKEN]]
        
        #lf_vocab_encodings.extend([self.gobal_syntax_vocabulary.get_stoi()[l] if l in self.gobal_syntax_vocabulary.get_stoi().keys() else self.gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN] for l in lftokens ])
        missing=False
        for iteridx, (toktype, tok) in enumerate(lftokens):
            if toktype in [ENTITY, RELATION, TYPE]:
                lf_sel.append(iteridx+1) # +1 for BOS_TOKEN, BOS_TOKEN is passed into the transformer model
                if tok in links:
                    nlf_map.append(links.index(tok))
                else:
                    nlf_map.append(links.index('MISSINGNODE'))
                    if SKIP_MISSING:
                        missing=True
                        break
                lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[toktype])
            # if token type is an action we will put real action index from vocab
            elif tok in gobal_syntax_vocabulary.get_stoi().keys():
                lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[tok])
            else:
                print(f'{tok} was not in the vocab.')
                lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN]) # why would this happen ?
            
        if missing and SKIP_MISSING:
            return None
        lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[EOS_TOKEN])
        lf_vocab_encodings = torch.tensor(lf_vocab_encodings, dtype=torch.int64)#, device=self.DEVICE)
        #lf_vocab_encodings = torch.cuda.LongTensor(lf_vocab_encodings)#, device=self.DEVICE)
        encoded_logical_form_list.append(lf_vocab_encodings)
        lf_selected_indices.append(lf_sel)
        node_lf_mappings.append(nlf_map)
        
    encoded_tensor = pad_sequence(encoded_logical_form_list, batch_first=True, padding_value=gobal_syntax_vocabulary.get_stoi()[PAD_TOKEN])
    return encoded_tensor, lf_selected_indices, node_lf_mappings

def binarize(inp_batch, is_train=TRAIN):
    '''
    list of dict_keys(['input', 'logical_form', 'gold_actions', 'conversation_triples', 'missing_element_list', 'fID'])
    '''
    
    gobalVocF = open('target_syntax.vocab')
    d = {}
    for l in gobalVocF.readlines():
        d[l.strip()] = 1
    gobal_syntax_vocabulary = vocab(Counter(d), specials=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, 'num', '0'])

    inputs = []
    logical_forms = []
    nodes = []
    node_type = []
    nodelinks = []
    extra_and_type_node_labels = []
    extra_and_type_links = []
    edges = []
    turnIDs=[]
    descriptions = []
    answers = []
    results = []
    qtypes = []
    node_encodings = []
    extra_node_encodings = []
    cum_num_nodes = []
    total_num_node=0
    combine=False
    #assert len(inp_batch) == 1
    
    for b in inp_batch:
        b = json.loads(b)
        if ANS_CONTEXT:
            inp1 = ' '.join(b['input'][:-1])
        else:
            inp1 = b['input'][0]
        
        inp2 = b['input'][-1]
        triples_link_list = b['conversation_triples']

        bnodes = []
        bnode_type = []
        blinks = []
        bedges = []
        bextra_and_type_node_labels =[]
        bextra_and_type_links = []
        
        if not is_train: # we do not want to train on empty elements
            if len(triples_link_list) == 0: triples_link_list.append([["random subject", "random relation", "random object"], ["Q", "P", "Q"]])
        for t in triples_link_list:
            triple, links = t[0], t[1]
            for iteridx, l in enumerate(links):
                if l not in blinks:
                    blinks.append(l)
                    bnodes.append(triple[iteridx])
                    # iter idx goes from 0-2, in order of sub rel obj
                    bnode_type.append(iteridx)
            
            sub_node_idx = blinks.index(links[0])
            rel_node_idx = blinks.index(links[1])
            obj_node_idx = blinks.index(links[2])
            # This is directed - do we want to make it undirected and add other edge too ?
            bedges.append([sub_node_idx, rel_node_idx])
            bedges.append([rel_node_idx, obj_node_idx])
        assert len(blinks) == len(bnodes)
        assert all([n in [0,1,2] for n in bnode_type]) , 'node types should be in 0 1 or 2 but got ' + str(bnode_type)
        ###if len(blinks) < 1 or len(bedges) < 1: continue    
        if len(blinks) < 1 or len(bedges) < 1:
            if is_train:
                continue
            else:
                print("This cannot happen")
                sys.exit(0)
        #bnodes.append('MISSINGNODE')
        #blinks.append('MISSINGNODE')
        
        for item in b['type_rel_label_dict'].items():
            bextra_and_type_links.append(item[0])
            bextra_and_type_node_labels.append(item[1])
        
        bextra_and_type_links.append('MISSINGNODE')
        bextra_and_type_node_labels.append('MISSINGNODE')
        
        bnode_type.append(3) # for missing node type
        turnID = b['turnID']
        turnIDs.append(turnID)
        qtypes.append(b['question-type'])
        descriptions.append(b['description'])
        inputs.append((inp1, inp2))
        logical_forms.append(b['gold_actions'])
        if not is_train:
            answers.append(b['answer'])
            results.append(b['results'])
        if combine:
            nodes.extend(bnodes)
            total_num_node += len(bnodes)
            cum_num_nodes.append(total_num_node)
        else:
            nodes.append(bnodes)
        
        nodelinks.append(blinks)
        node_type.append(bnode_type)
        extra_and_type_node_labels.append(bextra_and_type_node_labels)
        extra_and_type_links.append(bextra_and_type_links)

        #if len(bedges) > 0:
        bedges=torch.tensor(np.stack(bedges, axis=1), dtype=torch.long)
        #bedges=torch.cuda.LongTensor(np.stack(bedges, axis=1))
        edges.append(bedges)
    
    if inputs == []:
        return None
    input_encodings = tokenizer(inputs, truncation=True, padding=True,return_tensors='pt')
    if combine:
        node_encodings = tokenizer(nodes, truncation=True, padding=True,return_tensors='pt')
    else:
        for n in nodes:
            ne = tokenizer(n, truncation=True, padding=True,return_tensors='pt')
            node_encodings.append(ne)
    
    for n in extra_and_type_node_labels:
        ene = tokenizer(n, truncation=True, padding=True,return_tensors='pt')
        extra_node_encodings.append(ene)

    #padded_info = pad_and_encode_logical_form(logical_forms, nodelinks, tokenizer, gobal_syntax_vocabulary)
    #if padded_info == None:
    #    print(None)
    #    return None
    #encoded_logical_form, lf_selected_indices, node_lf_mappings = padded_info
    output_dict = {}
    output_dict['input_encodings'] = input_encodings
    output_dict['node_encodings'] = node_encodings
    output_dict['extra_node_encodings'] = extra_node_encodings
    output_dict['extra_and_type_node_labels'] = extra_and_type_node_labels
    output_dict['extra_and_type_links'] = extra_and_type_links
    output_dict['nodelinks'] = nodelinks
    output_dict['node_type'] = node_type
    output_dict['edges'] = edges
    output_dict['answers'] = answers
    output_dict['results'] = results
    #output_dict['encoded_logical_form'] = encoded_logical_form
    output_dict['logical_forms_str'] = logical_forms
    #output_dict['lf_selected_indices'] = lf_selected_indices
    #output_dict['node_lf_mappings'] = node_lf_mappings
    output_dict['cum_num_nodes'] = cum_num_nodes
    output_dict['inputs'] = inputs
    output_dict['turnIDs'] = turnIDs
    output_dict['question-types'] = qtypes
    output_dict['description'] =descriptions
    #return input_encodings, node_encodings, nodelinks, edges, encoded_logical_form, logical_forms, lf_selected_indices, node_lf_mappings, cum_num_nodes, fids
    #print(fids)
    return output_dict


###################################
def main(args):
    #nextlines_iter = itertools.islice(file_ptr, 100)
    print(args.keep_answer_context)
    global TRAIN, ANS_CONTEXT
    TRAIN=args.is_train
    ANS_CONTEXT=args.keep_answer_context
    inp_filename = args.inp_filename
    outname = args.outname
    def get_lines_iterator(filename, n):
        with open(filename) as fp:
            while True:
                lines = list(itertools.islice(fp, n))
                if lines:
                    yield lines
                else:
                    #raise StopIteration
                    break
    nextlines_iter = get_lines_iterator(inp_filename, args.batch_size)
    
    if 'train' in inp_filename:
        t=1368649/args.batch_size
    else:
        t=40023/args.batch_size
    dataset = []
    p_ct = 0
    if USE_POOL:
        pool = Pool(processes=args.n_cpus)
        print(pool)
        for processed_input in tqdm(pool.imap_unordered(binarize, nextlines_iter), total=t):  
            if processed_input is not None:
                dataset.append(processed_input.copy())
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:d}.pt".format(outname, p_ct)
                pt_file = os.path.join(args.save_path, pt_file)
                print(f'Saving shard: {pt_file}')
                with open(pt_file, 'w') as savefile:
                    # save.write('\n'.join(dataset))
                    torch.save(dataset, pt_file)
                    p_ct += 1
                    dataset = []
                    gc.collect()

        pool.close()
        pool.join()
    else:
        #for processed_input in tqdm(pool.imap_unordered(binarize, nextlines_iter), total=t):  
        for b in tqdm(nextlines_iter, total=t):
            processed_input = binarize(b)
            if processed_input is not None:
                dataset.append(processed_input.copy())
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:d}.pt".format(outname, p_ct)
                pt_file = os.path.join(args.save_path, pt_file)
                print(f'Saving shard: {pt_file}')
                with open(pt_file, 'w') as savefile:
                    # save.write('\n'.join(dataset))
                    torch.save(dataset, pt_file)
                    p_ct += 1
                    dataset = []
                    gc.collect()

    if (len(dataset) > 0):
        pt_file = "{:s}.{:d}.pt".format(outname, p_ct)
        pt_file = os.path.join(args.save_path, pt_file)
        print(f'Saving shard last: {pt_file}')
        with open(pt_file, 'w') as savefile:
            torch.save(dataset, pt_file)
            p_ct += 1
            dataset = []
           


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", required=True)
    parser.add_argument("-inp_filename", required=True)
    parser.add_argument("-outname", required=True)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument("-is_train", type=str2bool, nargs='?', const=True, default=False, required=True)
    parser.add_argument("-keep_answer_context", type=str2bool, nargs='?', const=True, default=True, required=False)

    parser.add_argument("-shard_size", default=10, type=int)
    parser.add_argument('-n_cpus', default=3, type=int)
    args = parser.parse_args()
    main(args)

