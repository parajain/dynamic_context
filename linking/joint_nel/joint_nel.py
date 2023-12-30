import json
import traceback
import os, re
from glob import glob
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import tqdm
import argparse
import pathlib
import ahocorasick
from os.path import exists
#from unidecode import unidecode
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sparql_helpers import SparqlHelper

ALLEN_NLP_TAG_WORDS="allen_tagged_words"



class NeighbourTypeLink:
    def __init__(self, json_kg_folder) -> None:
        print('Loading KG files ...')
        self.id_relation = json.loads(open(f'{json_kg_folder}/knowledge_graph/filtered_property_wikidata4.json').read())
        self.id_entity = json.loads(open(f'{json_kg_folder}/knowledge_graph/items_wikidata_n.json').read())
        self.TYPE_TRIPLES = json.loads(open(f'{json_kg_folder}/knowledge_graph/wikidata_type_dict.json').read())
        self.REV_TYPE_TRIPLES = json.loads(open(f'{json_kg_folder}/knowledge_graph/wikidata_rev_type_dict.json').read())
        self.TYPE_ID_LABEL = self._loadTypeIDLabelDict()

        print('Loading KG files done.')
        self.lemmatizer = WordNetLemmatizer()
        self.stops = set(stopwords.words('english'))
        self.not_found_type_count = 0
    
    def _loadTypeIDLabelDict(self):
        typeIDLabelDict = {}
        for k in self.TYPE_TRIPLES.keys():
            typeIDLabelDict[k] = self.id_entity[k]
        for k in self.REV_TYPE_TRIPLES.keys():
            typeIDLabelDict[k] = self.id_entity[k]
        return typeIDLabelDict
    
    def nltk_pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def lemmatize_sentence(self, sentence):
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        wordnet_tagged = map(lambda x: (x[0], self.nltk_pos_tagger(x[1])), nltk_tagged)
        lemmatized_sentence = []

        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    
    def lookup_types(self, utterance, typelist):
        utterance = self.lemmatize_sentence(utterance).split()
        filtered_types = []
        for tid in typelist:
            try:
                sval = self.TYPE_ID_LABEL[tid]
                #sval_words = sval.lower().split(' ')
                type = self.lemmatize_sentence(sval).split()
                type_non_stop = [w for w in type if w not in self.stops.union({'number'})] #number is a wd type but freq =count operator in utterances
                inter = (set(utterance) & set(type_non_stop))
                if type_non_stop and len(inter) == len(set(type_non_stop)):
                    filtered_types.append((tid, len(set(type_non_stop))))
            except KeyError:
                #print(f'Not found in type list {tid}')
                self.not_found_type_count += 1
            
        # This is what spice dataset paper does
        foo = [x for x in filtered_types if x!= ('Q2472587', 1)]
        if len(foo) == len(filtered_types)-1:
            foo.append(('Q502895', 2))
            filtered_types = foo
        filtered_types = [x for x in filtered_types if x!= ('Q528892', 1)]
        ## end dataset paper
        filtered_types = sorted(filtered_types, key=lambda x: x[1], reverse=True)
        types = filtered_types
        if types:
            keep = [(self.TYPE_ID_LABEL[types[0][0]], types[0][0])]
            for t, _ in types[1:]:
                present = False
                for l, _ in keep:
                    incl = set(self.TYPE_ID_LABEL[t].split()).intersection(set(l.split()))
                    if len(incl) == len(set(self.TYPE_ID_LABEL[t].split())):
                        present = True
                        break
                if not present:
                    keep.append((self.TYPE_ID_LABEL[t], t))
        filtered_types = [t for t,_ in types if t in [k for _, k in keep]]
            


        return list(filtered_types)



    def lookup_types_simple(self, utterance, typelist):
        utterance_word  = utterance.lower().split(' ')
        
        filtered_types = set()
        for tid in typelist:
            try:
                sval = self.TYPE_ID_LABEL[tid]
                sval_words = sval.lower().split(' ')
                for w in sval_words: # this is also matching stop words like 'of'
                    if w in utterance_word:
                        filtered_types.add(tid)
            except KeyError:
                self.not_found_type_count += 1

        return list(filtered_types)
    

class JointLinking:
    def __init__(self, automaton_file, json_kg_folder) -> None:
        self.sparql_helper = SparqlHelper(load_cache=True)
        self.neighbour_link_types = NeighbourTypeLink(json_kg_folder)
        print('Loading automaton..')
        self.automaton = pickle.load(open(automaton_file, 'rb'))
        print('Loading automaton done..')
        self.total_entity_f1 = 0
        self.total_entity_prec = 0
        self.total_entity_recall = 0
        self.total_type_f1 = 0
        self.total_type_prec = 0
        self.total_type_recall = 0
        self.num_instances = 0
        self.entity_not_present = 0
        

    def get_allennlp_entities(self, turn):
        if ALLEN_NLP_TAG_WORDS not in turn.keys():
            return None
        tagged_words = turn[ALLEN_NLP_TAG_WORDS]
        return tagged_words
    
    def link_entities(self, words):
        def preprocess(text):
            text = text.lower()
            return text
        
        ent_ids = set()
        for w in words:
            selected_id = None
            mlen = -1
            w = preprocess(w)
            for end_index, (id, strval) in self.automaton.iter_long(w):
                if len(strval) > mlen:
                    selected_id = id
                    mlen = len(strval)
            if selected_id is None:
                #print(f'Entity not found in automaton {w}')
                pass
            else:
                ent_ids.add(selected_id)
        
        return ent_ids
            
    def get_ent_subgraph(self, turn):
        local_subgraph = turn['local_subgraph']
        local_subgraph_nel = turn['local_subgraph_nel']
        
    def check_links(self, ent_ids, turn):
        local_subgraph = turn['local_subgraph']
        local_subgraph_nel = turn['local_subgraph_nel']
        print(len(ent_ids) - (len(local_subgraph.keys()) + len(local_subgraph_nel)) ) # mostly -ve
        for eid in ent_ids:
            if eid not in local_subgraph and eid not in local_subgraph_nel:
                self.entity_not_present += 1
                print('Not in graph ', eid)
                return False
        
    def link_json(self, param):
        try:
            self._link_json(param)
        except:
            print('Failed ', param)
            return None

    def _link_json(self, param):
        input_jsonfile, output_jsonfile = param 
        conversation = json.load(open(input_jsonfile, 'r'))
        turns = len(conversation) // 2
        is_clarification = False
        conversation_level_gold_etities = set()
        conversation_level_tagged_etities = set()
        conversation_level_gold_types = set()
        conversation_level_tagged_types = set()

        turn_types_from_graph = set()
        type_triples = {}
        

        prev_system=None
        prev_user=None
        for i in range(turns):
            if is_clarification:
                is_clarification = False
                continue
            user = conversation[2 * i]
            system = conversation[2 * i + 1]
            if 'context' in user.keys():
                del user['context']
            if 'context' in system.keys():
                del system['context']
            
            if user['question-type'] == 'Clarification':
                
                is_clarification = True
                next_user = conversation[2 * (i + 1)]
                next_system = conversation[2 * (i + 1) + 1]
                
                if 'context' in next_user.keys():
                    del next_user['context']
                if 'context' in next_system.keys():
                    del next_system['context']
                
                allen_tagged_words = []
                twuser = self.get_allennlp_entities(user)
                if twuser is not None:
                    allen_tagged_words.extend(twuser)
                tw_next_user = self.get_allennlp_entities(next_user)
                if tw_next_user is not None:
                    allen_tagged_words.extend(tw_next_user)
                
                entity_links = self.link_entities(allen_tagged_words)
                if prev_system is not None:
                    if 'entities_in_utterance' in user.keys():
                        prev_ans_entities = set(user['entities_in_utterance'])
                    else:
                        prev_ans_entities = set(user['entities'])
                    # This is because prior work use gold previous answers.
                    entity_links.update(prev_ans_entities)
                    

                # This was local we can do this globally, since we are  going to filter.
                #turn_types_from_graph = set()
                #type_triples = {}
                for eid in entity_links:
                    t, triples = self.sparql_helper.get_2hop_types_triples(eid)
                    turn_types_from_graph.update(t)
                    #assert any([k not in type_triples.keys() for k in triples.keys() ])
                    for k in triples.keys():
                        if k in type_triples:
                            #print('exist ', k)
                            type_triples[k].update(triples[k])
                        else:
                            type_triples[k] = set(triples[k])
                
                filtered_turn_types = set()
                ft_user = self.neighbour_link_types.lookup_types(user['utterance'], turn_types_from_graph)
                ft_next_user = self.neighbour_link_types.lookup_types(next_user['utterance'], turn_types_from_graph)
                if ft_user:
                    filtered_turn_types.update(ft_user)
                if ft_next_user:
                    filtered_turn_types.update(ft_next_user)
                
                filtered_type_triples = set()
                for t in filtered_turn_types:
                    if t == 'Q502895' and 'Q2472587' in type_triples:
                        #filtered_type_triples[t] = type_triples['Q2472587']
                        filtered_type_triples.update(type_triples['Q2472587'])
                    else:
                        #filtered_type_triples[t] = type_triples[t]
                        filtered_type_triples.update(type_triples[t])
                


                next_user['joint_type_links'] = list(filtered_type_triples)
                next_user['joint_entity_links'] = list(entity_links)

                user_tags = entity_links
                if 'entities_in_utterance' in user.keys():
                    user_gold = set(user['entities_in_utterance'])
                else:
                    user_gold = set(user['entities'])
                

                # because next turn will be skipped and will become previous
                prev_user = next_user
                prev_system = next_system



            else: # else not clarification
                allen_tagged_words = self.get_allennlp_entities(user)
                if allen_tagged_words is None:
                    continue
                    
                entity_links = self.link_entities(allen_tagged_words)
                if prev_system is not None:
                    if 'entities_in_utterance' in user.keys():
                        prev_ans_entities = set(user['entities_in_utterance'])
                    else:
                        prev_ans_entities = set(user['entities'])
                    # This is because prior work use gold previous answers.
                    entity_links.update(prev_ans_entities)

                # Keeping this globally for each interaction
                #turn_types_from_graph = set()
                #type_triples = {}
                for eid in entity_links:
                    t, triples = self.sparql_helper.get_2hop_types_triples(eid)
                    turn_types_from_graph.update(t)
                    for k in triples.keys():
                        if k in type_triples:
                            type_triples[k].update(triples[k])
                        else:
                            type_triples[k] = set(triples[k])
                
                filtered_turn_types = set()
                ft_user = self.neighbour_link_types.lookup_types(user['utterance'], turn_types_from_graph)
                if ft_user:
                    filtered_turn_types = set(ft_user)
                
                filtered_type_triples = set()
                for t in filtered_turn_types:
                    if t == 'Q502895' and 'Q2472587' in type_triples:
                        filtered_type_triples.update(type_triples['Q2472587'])
                    else:
                        filtered_type_triples.update(type_triples[t])

                user['joint_type_links'] = list(filtered_type_triples)
                user['joint_entity_links'] = list(entity_links)


                user_tags = entity_links
                if 'entities_in_utterance' in user.keys():
                    user_gold = set(user['entities_in_utterance'])
                else:
                    user_gold = set(user['entities'])

                prev_user = user
                prev_system = system

        

        if not os.path.isdir(os.path.dirname(output_jsonfile)):
            os.makedirs(os.path.dirname(output_jsonfile))
        json.dump(conversation, open(output_jsonfile, 'w'), ensure_ascii=False, indent=2)
    

    def process(self, args_list, n_parallel):
        pool = ThreadPool(n_parallel)
        for processed_filename in tqdm.tqdm(pool.imap_unordered(self.link_json, args_list), total=len(args_list)):
            pass


############################### MAIN #######################            
def process_files_parallel(args_list, kg_folder, colour=None):
    print('Process files ...')
    wordnet.ensure_loaded()
    joint_linking = JointLinking(automaton_file='automaton.pkl', json_kg_folder=kg_folder)
    
    a_lst = [(ifile, ofile) for ifile, ofile in args_list]
    joint_linking.process(a_lst, args.n_parallel)
    joint_linking.sparql_helper.save_cache()
    import sys
    sys.exit(0)
    #pool = ThreadPool(args.n_parallel)
    pool = Pool(args.n_parallel)
    outfiles = set()
    for processed_filename in tqdm.tqdm(pool.imap_unordered(joint_linking.link_json, a_lst), total=len(a_lst)):
        #if processed_filename in outfiles:
        #    print('processed_filename', processed_filename)
        outfiles.add(processed_filename)
    joint_linking.sparql_helper.save_cache()


def process_files(args_list, kg_folder, colour=None):
    print('Process files ...')
    joint_linking = JointLinking(automaton_file='automaton.pkl', json_kg_folder=kg_folder)
    
    a_lst = [(ifile, ofile) for ifile, ofile in args_list]
    outfiles = set()
    for param in tqdm.tqdm(a_lst):
        processed_filename = joint_linking.link_json(param)
        #outfiles.add(processed_filename)
    joint_linking.sparql_helper.save_cache()
    '''
    f1 = joint_linking.total_entity_f1 / joint_linking.num_instances
    prec = joint_linking.total_entity_prec / joint_linking.num_instances
    rec = joint_linking.total_entity_recall / joint_linking.num_instances
    print('F1 score entity ', f1, prec, rec)
    typef1 = joint_linking.total_type_f1 / joint_linking.num_instances
    typeprec = joint_linking.total_type_prec / joint_linking.num_instances
    typerec = joint_linking.total_type_recall / joint_linking.num_instances
    print('F1 score type ', typef1, typeprec, typerec)
    print('entity_not_present ', joint_linking.entity_not_present)
    '''


def tag_str_all_files(args):
    '''
    after some issues in data were fixed some more files could be tagged and added
    '''
    allfiles = []
    #for f in open(args.file_path, 'r').readlines():
    #    allfiles.append(f.rstrip())
    #if args.end > 0:
    #    allfiles = allfiles[args.start:args.end]
    #print('Processing from {s} and {e}, will save at {p}'.format(s=args.start,e=args.end,p=args.save_path))
    print(f'Reading json files list from {args.data_path}/{args.split}')
    allfiles = glob(f'{args.data_path}/{args.split}/*' + '/*.json')
    existing_out_files = glob(f'{args.save_path}/{args.split}/*' + '/*.jointlinking')
    existing_out_files = set(existing_out_files)
    #print(allfiles[:3])
    
    #args_list = [(inp_file, os.path.join(args.save_path, corpus_type, os.path.basename(inp_file))) for inp_file in allfiles]

    def get_inp_out_file(filename, local_path, out_dir):
        lsys=args.data_path
        inp_file = filename
        outfile = filename.replace(lsys, out_dir) + '.jointlinking'
        if outfile in existing_out_files:
            print('Skip ', outfile)
            return None
        return (inp_file, outfile)
    args_list = []
    for inp_file in allfiles:
        i_o_file = get_inp_out_file(inp_file, args.data_path, args.save_path)
        if i_o_file:
            args_list.append(i_o_file)



    #args_list = [get_inp_out_file(inp_file, args.data_path, args.save_path) for inp_file in allfiles]
    #print(args_list)
    #process_files_parallel(args_list, kg_folder=args.kg_folder)
    process_files(args_list, kg_folder=args.kg_folder)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    #python joint_nel.py -data_path /home/s1959796/csqparsing/condata/spice_dataset/CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned_nelg2 -save_path temp_save -split valid -kg_folder /home/s1959796/csqparsing/condata/spice_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=False)
    parser.add_argument("-save_path", required=True)
    parser.add_argument("-kg_folder", required=True)
    #parser.add_argument("-file_path", required=False)
    
    parser.add_argument('-split', required=True)
    parser.add_argument('-n_parallel', default=15, type=int)
    #parser.add_argument('-start', default=10, type=int)
    #parser.add_argument('-end', default=10, type=int)
    parser.add_argument('-debug', nargs='?',const=False,default=False)
    args = parser.parse_args()
    tag_str_all_files(args)
