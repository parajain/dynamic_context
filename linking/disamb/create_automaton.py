import json
import traceback
import os, re
from glob import glob
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import argparse
import pathlib
import ahocorasick
from os.path import exists
from unidecode import unidecode
import pickle

basic_stops = ['where', 'did', 'how', 'many', 'where', 'when', 'which']

def count_file(data_file):
    try:
        return _getcounts(data_file)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        return 0

def _getcounts(data_file):
    
    try:
        data = json.load(open(data_file, 'r'))
    except json.decoder.JSONDecodeError as e:
        print('Failed loading json file: ', data_file)
        raise e
   
    entity_counts = {}
    def add_to_dict(e):
        if e in entity_counts.keys():
            entity_counts[e] += 1
        else:
            entity_counts[e] = 1
    
    for conversation in [data]:
        
        turns = len(conversation) // 2

        for i in range(turns):            
            
            user = conversation[2*i]
            system = conversation[2*i + 1]
            if 'entities_in_utterance' in user.keys() or 'entities' in user.keys():
                if 'entities_in_utterance' in user.keys():
                    user_gold = user['entities_in_utterance']
                    for e in user_gold:
                        add_to_dict(e)
                else:
                    user_gold = set(user['entities'])
                    for e in user_gold:
                        add_to_dict(e)
                
            #   print(user['utterance'], data_file)


            if 'entities_in_utterance' in system.keys():
                system_gold = set(system['entities_in_utterance'])
                for e in system_gold:
                    add_to_dict(e)
            else:
                print('entities_in_utterance not present', data_file)

    return entity_counts

def get_count(args):
    global_entity_counts= {}
    def add_to_global_dict(e, c):
        if e in global_entity_counts.keys():
            global_entity_counts[e] += c
        else:
            global_entity_counts[e] = c
    
    train_path = args.data_path + '/train/*'
    train_files = glob(train_path + '/*.json')
    print('Train files ', train_path, len(train_files))
    corpora = {'train': train_files}

    for corpus_type in corpora.keys():
        filelist = corpora[corpus_type]
        pool = Pool(args.n_cpus)
        for entity_count in tqdm(pool.imap_unordered(count_file, filelist), total=len(filelist)):
            for e, c in entity_count.items():
                add_to_global_dict(e, c)

        pool.close()
        pool.join()        
    json.dump(global_entity_counts, open('entity_count.json', 'w', encoding='utf8'), indent=2, ensure_ascii=False)
    return global_entity_counts


def disambiguate(listofids, entity_count):
    counts = []
    for id in listofids:
        c=entity_count.get(id, -1)
        #if c > -1:
        #    print(id)
        counts.append(c)
    max_id = counts.index(max(counts))
    return listofids[max_id]

def preprocess(text):
  text = text.lower()
  text = text.translate(str.maketrans('', '', ",.?"))
  text = ' '.join([t  for t in text.split() if t not in basic_stops])
  return text.lower()

def create_index(kg_path, file_list, entity_count):
    index = ahocorasick.Automaton()
    for filename in file_list:
        fpath = os.path.join(kg_path, filename)
        print('Loading json file from ', fpath)
        id_val_dict = json.load(open(fpath, 'r'))
        count = 0
        for val, idlist in tqdm(id_val_dict.items(), total=len(id_val_dict)):
            disambiguated_id = disambiguate(idlist, entity_count)
            index.add_word(preprocess(val), (disambiguated_id, val))
            count += 1
        print(f'Added {count} items.')
    
    index.make_automaton()
    return index


def save_automaton(entity_count, kg_path):
    automaton_filename = 'automaton.pkl'
    #kg_path='wikidata_proc_json/wikidata_proc_json_2/'
    file_list = ['items_wikidata_n.json.redump']
    automaton = create_index(kg_path=kg_path, file_list=file_list, entity_count = entity_count)
    pickle.dump(automaton, open(automaton_filename, 'wb'))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-kg_path", required=True)
    parser.add_argument('-data_path', required=True)
    parser.add_argument('-n_cpus', default=10, type=int)
    args = parser.parse_args()
    entity_count = get_count(args)
    save_automaton(entity_count)
