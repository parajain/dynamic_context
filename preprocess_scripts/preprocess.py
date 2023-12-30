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
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

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
##
TURN_ID = 'turnID'
USER = 'USER'
SYSTEM = 'SYSTEM'

##
TGOLD = 'gold'
TLINKED = 'linked'
TJOINT_LINKED = 'joint_link'
TNONE = 'none'
NEGOLD = 'gold'
NELGNEL = 'lgnel'
NEALLENNEL = 'allennel'
JOINTNEL = 'jointnel'
NESTRNEL = 'strnel'
ES_LINKS = 'es_links'
SPANS = 'tagged_words'
ALLEN_SPANS = 'allen_tagged_words'
ALLEN_TAGS = 'allennlp_tags'
ALLEN_ES_LINKS = 'allen_es_links'
JOINT_ES_LINKS = 'joint_entity_links'
JOINT_TYPE_LINKS= 'joint_type_links'
STR_ES_LINKS = 'str_es_links'
STR_SPANS = 'str_tagged_words'

##
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


kb_graph='spice_dataset/knowledge_graph'
TYPE_TRIPLES = json.loads(open(kb_graph +  '/wikidata_type_dict.json').read())
REV_TYPE_TRIPLES = json.loads(open(kb_graph +  '/wikidata_rev_type_dict.json').read())
ID_RELATION = json.loads(open(kb_graph +  '/filtered_property_wikidata4.json').read())
ID_ENTITY = json.loads(open(kb_graph +  '/items_wikidata_n.json').read())
        

def get_kb_elements_in_current_triples(triples):
    ele = []
    for t in triples:
        ele.extend(list(t[1]))
    ele = set(ele)
    #print(list(ele)) 
    return ele

def get_type_rel_dict(gold_types, existing_ele):
    def getTypesGraph(gold_types):
        tgraph = {}
        for t in gold_types:
            # just take the type and associated relations
            try:
                tg = [rel for rel in TYPE_TRIPLES[t].keys()]
            except KeyError:
                tg = []

            try:
                tg.extend([rel for rel in REV_TYPE_TRIPLES[t].keys()])
            except KeyError:
                pass

            tgraph[t] = list(set(tg))

        return tgraph
    
    def extendIdsMap(types_graph):
        ids_map = {}
        for t_head in types_graph.keys():
            if t_head not in ids_map.keys() and t_head not in existing_ele:
                ids_map[t_head] = ID_ENTITY[t_head]

            for r in types_graph[t_head]:
                if r not  in ids_map.keys() and r not in existing_ele:
                    ids_map[r] = ID_RELATION[r]
                #if r in existing_ele:
                #    print(' ==========================skip ', r)

        return ids_map
    
    return extendIdsMap(getTypesGraph(gold_types))


def format_delex(turn_sp):
    # reformat as Lasagna expects the logical forms (lists of lists with token types)
    lasagneFormat = []
    if 'sparql' not in turn_sp.keys():
        print(turn_sp['utterance'])
    sparqToks = turn_sp['sparql'].replace('.', ' . ').replace('wd:', 'wd: ') \
        .replace('wdt:', 'wdt: ').replace('{', ' { ').replace('}', ' } ') \
        .replace('(', ' ( ').replace(')', ' ) ') \
        .split()
    sparqdlToks = turn_sp['sparql_delex'].replace('.', ' . ') \
        .replace('{', ' { ').replace('}', ' } ') \
        .replace('(', ' ( ').replace(')', ' ) ') \
        .split()
    assert len(sparqToks) == len(sparqdlToks), f
    'Different len in tokens sparqToks={len(sparqToks)} vs sparqdlToks={len(sparqdlToks)}'
    for sparq, sparqdl in zip(sparqToks, sparqdlToks):
        if sparqdl == ENTITY or sparqdl == RELATION or sparqdl == TYPE or sparqdl == VALUE:
            lasagneFormat.append([sparqdl, sparq])
        else:
            lasagneFormat.append([ACTION, sparqdl])

    return lasagneFormat


def get_kb_elements(actions):
        kbelements= set()
        for action in actions:
            assert action[0] in [RELATION, TYPE, ENTITY, ACTION, VALUE], 'We might have decided to change the format again, actions received : ' + action[0] + ' full ' + str(actions)
            if action[0] in [RELATION, TYPE, ENTITY]:
                kbelements.add((action[1], action[0]))
        return kbelements


def detokenize(tokens):
    def is_subtoken(word):
        if word[:2] == "##":
            return True
        else:
            return False

    #tokens = ['why', 'isn', "##'", '##t', 'Alex', "##'", 'text', 'token', '##izing']
    #tokens = 'nik ##os lia ##ko ##poulos'.split()
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            restored_text.append(tokens[i] + tokens[i+1][2:])
            if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return ' '.join(restored_text)



def lineariseAsTriples(subgraph):
    label_triples = []
    wikidata_entity_triples = []
    entity_label_dict = {}
    entity_type_dict = {}
    all_entity_set = set()
    all_entity_relation_set = set()
    all_rel_set = set()
    all_type_set = set()
    edge_set = set()
    all_ent_rel_types_text = []

    
    
    
    def format_as_triple(s, r, o, as_string=True):
        if as_string:
            return ' {RELARGS_DELIMITER_BERT} '.join([s, r, o]).format(RELARGS_DELIMITER_BERT=RELARGS_DELIMITER_BERT)
        else:
            if '##' in s:
                s = detokenize(s.split())
            if '##' in r:
                r = detokenize(r.split())
            if '##' in o:
                o = detokenize(o.split())
            return (s, r, o)

    for parent_e in subgraph.keys():
        pg = subgraph[parent_e]
        all_entity_set.add(parent_e)
        plabel = pg['label']

        # ent_rel_types_text.add(plabel)
        # Why is there a list of list can have multiple types ?
        assert len(pg['type']) < 2, pg['type']
        ent_rel_types_text = []
        ent_rel_types_text.append(plabel)
        if (len(pg['type'])) > 1:  # this is because we may not have type annotation here
            # print(subgraph)
            ptype, ptype_label = pg['type'][0][0], pg['type'][0][1]
            all_type_set.add(ptype)
            if parent_e not in entity_label_dict.keys():
                entity_label_dict[parent_e] = plabel
                entity_type_dict[parent_e] = [ptype, ptype_label]

            if ptype not in entity_label_dict.keys():
                entity_label_dict[ptype] = ptype_label
            label_triple = format_as_triple(
                plabel, 'type', ptype_label, as_string=False)

            
            e_triple = format_as_triple(
                parent_e, 'P31', ptype, as_string=False)
            edge_set.add(e_triple[0] + e_triple[1])
            edge_set.add(e_triple[1] + e_triple[2])
            label_triples.append((label_triple, e_triple))
            wikidata_entity_triples.append(' '.join(e_triple))

            ent_rel_types_text.append(ptype_label)
        all_ent_rel_types_text.append(' [SEP] '.join(list(ent_rel_types_text)))
        # there wa a bug obj is subject and sub or ibj
        #subjects_g = pg['subject']
        subjects_g = pg['object']
        for rel in subjects_g:
            ent_rel_types_text = []
            ent_rel_types_text.append(plabel)
            # if (parent_e, rel) in all_entity_relation_set:
            #    print('Exitt ', parent_e, rel)
            all_entity_relation_set.add(('Obj', parent_e, rel))
            all_rel_set.add(rel)
            rel_label = subjects_g[rel]['label']
            ent_rel_types_text.append(rel_label)
            if rel not in entity_label_dict.keys():
                entity_label_dict[rel] = rel_label
            # why list of list here?
            #ob, ob_label = subjects_g[rel]['type_restriction'][0][0], subjects_g[rel]['type_restriction'][0][1]
            for ob, ob_label in subjects_g[rel]['type_restriction']:
                all_type_set.add(ob)
                ent_rel_types_text.append(ob_label)
                if ob not in entity_label_dict.keys():
                    entity_label_dict[ob] = ob_label
                    entity_type_dict[ob] = [ob, ob_label]
                label_triple = format_as_triple(
                    plabel, rel_label, ob_label, as_string=False)
                e_triple = format_as_triple(parent_e, rel, ob, as_string=False)
                edge_set.add(e_triple[0] + e_triple[1])
                edge_set.add(e_triple[1] + e_triple[2])
                label_triples.append((label_triple, e_triple))
                wikidata_entity_triples.append(' '.join(e_triple))
            all_ent_rel_types_text.append(
                ' [SEP] '.join(list(ent_rel_types_text)))

        # there wa a bug obj is subject and sub or ibj
        #objects_g = pg['object']
        objects_g = pg['subject']
        for rel in objects_g:
            ent_rel_types_text = []
            ent_rel_types_text.append(plabel)
            # if (parent_e, rel) in all_entity_relation_set:
            #    print('Exitt ', parent_e, rel)
            all_entity_relation_set.add(('sub', parent_e, rel))
            all_rel_set.add(rel)
            rel_label = objects_g[rel]['label']
            ent_rel_types_text.append(rel_label)
            if rel not in entity_label_dict.keys():
                entity_label_dict[rel] = rel_label
            #ob, ob_label = objects_g[rel]['type_restriction'][0][0], objects_g[rel]['type_restriction'][0][1]
            for ob, ob_label in objects_g[rel]['type_restriction']:
                all_type_set.add(ob)
                ent_rel_types_text.append(ob_label)
                if ob not in entity_label_dict.keys():
                    entity_label_dict[ob] = ob_label
                    entity_type_dict[ob] = [ob, ob_label]
                label_triple = format_as_triple(
                    ob_label, rel_label, plabel, as_string=False)
                e_triple = format_as_triple(ob, rel, parent_e, as_string=False)
                edge_set.add(e_triple[0] + e_triple[1])
                edge_set.add(e_triple[1] + e_triple[2])
                label_triples.append((label_triple, e_triple))
                wikidata_entity_triples.append(' '.join(e_triple))
            all_ent_rel_types_text.append(
                ' [SEP] '.join(list(ent_rel_types_text)))

    parent_entities = subgraph.keys()
    # return label_triples, wikidata_entity_triples, parent_entities, entity_label_dict, entity_type_dict
    # all_entity_set, all_rel_set, all_type_set, edge_set, all_ent_rel_types_text, all_entity_relation_set
    return label_triples, all_entity_set, all_rel_set, all_type_set



def get_subgraph(turn_nels, local_subgraph, local_subgraph_nel):
    graph = {}
    for ne in turn_nels:
        if ne in local_subgraph.keys():
            graph[ne] = local_subgraph[ne]
        if local_subgraph_nel and ne in local_subgraph_nel.keys():
            graph[ne] = local_subgraph_nel[ne]

    return graph

def take_nels(nel_field):
    ret = []
    if len(nel_field) > 0:
        if isinstance(nel_field[0], list):
            ret = [x[0] for x in nel_field if
                    len(x) > 0]  # TODO: take the top one, see if we want to choose other top-k
        else:
            ret = nel_field
    return ret

def get_non_gold_nel(nentities, turn):
    '''Adds context: textual and Named Entity annotations. NER annotations could
    be Lasagne ones or external NER tool (es_links).'''
    ret_nel = []

    if turn['speaker'] == USER:
        if nentities == NEALLENNEL: # AllenNLP -based NEL annotations
            ret_nel = take_nels(turn[ALLEN_ES_LINKS])
        elif nentities == JOINTNEL:
            if JOINT_ES_LINKS in turn.keys():
                ret_nel = take_nels(turn[JOINT_ES_LINKS])
        elif nentities == NESTRNEL: # Str -based NEL annotations
            ret_nel = take_nels(turn[STR_ES_LINKS])
    else: # is SYSTEM
        ret_nel = turn['entities_in_utterance'] # as previous works take previous gold answer (systems output answers so thye know the entity they output -> no need to do NEL)

    return ret_nel

def get_type_triples(turn):
    triples = turn[JOINT_TYPE_LINKS]
    triples_with_values = []
    for triple in triples:
        try:
            # [['wang jingwei', 'family name', 'family name'], ['Q22303', 'P734', 'Q101352']]
            triples_with_values.append(((ID_ENTITY[triple[0]], ID_RELATION[triple[1]], ID_ENTITY[triple[2]]), tuple(triple)))
        except KeyError:
            continue
    return triples_with_values
        
    


def preprocess_file(inp_args):
    data_file, args = inp_args
    try:
        return _preprocess_file(args, data_file)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        raise e


def _preprocess_file(args, data_file):
    history_conversation_triples =[]
    history_all_entity = []
    history_all_rel = []
    history_all_type = []
    input_data = []
    last_n = args.last_n
    nentities = args.nentities
    #include_ans_context = args.include_ans_context

    try:
        data = json.load(open(data_file, 'r'))
    except json.decoder.JSONDecodeError as e:
        print('Failed loading json file: ', data_file)
        raise e
    fID = data_file.split('/')
    fID = f'{fID[-3]}#{fID[-2]}#{fID[-1].split(".json")[0]}'
    for conversation in [data]:
        is_clarification = False
        prev_user_conv = None
        prev_system_conv = None
        turns = len(conversation) // 2

        for i in range(turns):
            input = []
            logical_form = []
            turn_nels = []
            # If the previous was a clarification question we basically took next 
            # logical form so need to skip
            if is_clarification:
                is_clarification = False
                continue
            user = conversation[2*i]
            system = conversation[2*i + 1]
            description = user['description']
            if user['question-type'] == 'Clarification':
                is_clarification = True
                next_user = conversation[2*(i+1)]
                next_system = conversation[2*(i+1) + 1]
                input.append(prev_user_conv['utterance'])
                input.append(prev_system_conv['utterance']) # 
                input.append(user['utterance'])
                input.append(system['utterance']) # it is fine to take system here because this is clarification question and parse is basically next_system
                #types = user['type_list'] if 'type_list' in user.keys() else []
                
                if args.types_to_use == TGOLD:
                    context_types = prev_user_conv['type_list'] if 'type_list' in prev_user_conv.keys() else []
                elif args.types_to_use == TLINKED:
                    # these are automatically linked types
                    if prev_system_conv and 'type_subgraph' in prev_system_conv.keys():
                        context_types = prev_system_conv['type_subgraph']
                    else: context_types = {}
                    #existing_types = len(types.keys()) > 0
                
                types = None
                if args.types_to_use == TGOLD:
                    types = user['type_list'] if 'type_list' in user.keys() else []
                    types = list(set(context_types + types))
                    existing_types = len(types) > 0
                elif args.types_to_use == TLINKED and 'type_subgraph' in next_system.keys():
                    # these are automatically linked types
                    types = next_system['type_subgraph']
                    for k, v in context_types.items(): types[k] = v
                    existing_types = len(types.keys()) > 0

                #subgraph = system['local_subgraph'] -- to do this needs to be added too
                #because this one is clarification we basically skipping to 
                # the next question and including more this one as context
                input.append(next_user['utterance'])
                if nentities == NEGOLD:
                    subgraph = next_system['local_subgraph']
                else:
                    turn_nels.extend(get_non_gold_nel(nentities, prev_user_conv))
                    turn_nels.extend(get_non_gold_nel(nentities, prev_system_conv))
                    turn_nels.extend(get_non_gold_nel(nentities, user))
                    turn_nels.extend(get_non_gold_nel(nentities, system))
                    turn_nels.extend(get_non_gold_nel(nentities, next_user))
                    local_subgraph_nel = next_system['local_subgraph_nel'] if 'local_subgraph_nel' in next_system.keys() else {}
                    subgraph = get_subgraph(turn_nels, next_system['local_subgraph'], local_subgraph_nel)
                    ## add type graph here
                
                triples, all_entity_set, all_rel_set, all_type_set = lineariseAsTriples(subgraph)
                if args.types_to_use == TJOINT_LINKED:
                    assert types == None, 'we should not have extracted types from other NEL when we are using joint linking'
                    #ToDo: we can include type from prev user
                    type_triples = get_type_triples(next_user)
                    triples.extend(type_triples)
                #else:
                #    print('Not using ', args.types_to_use)

                if 'sparql' not in next_system.keys():
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                    #print('no sparql')
                    continue
                    
                gold_actions = format_delex(next_system)

                # track context history
                prev_user_conv = next_user.copy()
                prev_system_conv = next_system.copy()
                results = next_system['sparql_entities'] if 'sparql_entities' in next_system.keys() \
                                else next_system['all_entities']
                answer = next_system['utterance']
            else:
                if i != 0:
                    input.append(prev_user_conv['utterance'])
                    input.append(prev_system_conv['utterance'])
                input.append(user['utterance'])
                #subgraph = system['local_subgraph']
                if nentities == NEGOLD:
                    subgraph = system['local_subgraph']
                else:
                    if i != 0:
                        turn_nels.extend(get_non_gold_nel(nentities, prev_user_conv))
                        turn_nels.extend(get_non_gold_nel(nentities, prev_system_conv))
                    turn_nels.extend(get_non_gold_nel(nentities, user))
                    local_subgraph_nel = system['local_subgraph_nel'] if 'local_subgraph_nel' in system.keys() else {}
                    subgraph = get_subgraph(turn_nels, system['local_subgraph'], local_subgraph_nel)
                    ## add type graph here
                #types = user['type_list'] if 'type_list' in user.keys() else []
                if args.types_to_use == TGOLD:
                    context_types = prev_user_conv['type_list'] if 'type_list' in prev_user_conv.keys() else []
                elif args.types_to_use == TLINKED:
                    # these are automatically linked types
                    if prev_system_conv and 'type_subgraph' in prev_system_conv.keys():
                        context_types = prev_system_conv['type_subgraph']
                    else: context_types = {}
                
                types = None
                existing_types = True # if we decide to not include types, then are fine
                if args.types_to_use == TGOLD:
                    types = user['type_list'] if 'type_list' in user.keys() else []
                    types = list(set(context_types + types))
                    existing_types = len(types) > 0
                elif args.types_to_use == TLINKED and 'type_subgraph' in system.keys():
                    # these are automatically linked types
                    types = system['type_subgraph']
                    for k,v in context_types.items(): types[k] = v
                    
                triples, all_entity_set, all_rel_set, all_type_set = lineariseAsTriples(subgraph)
                if args.types_to_use == TJOINT_LINKED:
                    assert types == None, 'we should not have extracted types from other NEL when we are using joint linking'
                    #ToDo: we can include type from prev user
                    type_triples = get_type_triples(user)
                    triples.extend(type_triples)

                if 'sparql' not in system.keys():
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()
                    print('no sparql')
                    continue
                gold_actions = format_delex(system)

                # track context history
                prev_user_conv = user.copy()
                prev_system_conv = system.copy()
                
                answer = system['utterance']
                results = system['sparql_entities'] if 'sparql_entities' in system.keys() else system['all_entities']
                question_type = user['question-type']
                if question_type in ['Quantitative Reasoning (Count) (All)',
                                            'Comparative Reasoning (Count) (All)']:
                    if 'sparql_entities' in system.keys() :
                        answer = f'{len(system["sparql_entities"])}'
                    elif len(set(system["all_entities"])) > 0:
                        # redefine answer value as some gold sets have duplicates
                        # Lasagne annotation tool always takes set(gold) in all comparisons to assess formula adequacy
                        answer = f'{len(set(system["all_entities"]))}'
            
            for action in gold_actions:
                if action[0] == ACTION:
                    logical_form.append(action[1])
                elif action[0] == RELATION:
                    logical_form.append(RELATION)
                elif action[0] == TYPE:
                    logical_form.append(TYPE)
                elif action[0] == ENTITY:
                    logical_form.append(ENTITY)
                elif action[0] == VALUE:
                    logical_form.append(action[0])
                else:
                    raise Exception(f'Unkown logical form action {action[0]}')

            history_conversation_triples.append(triples)
            history_all_entity.append(all_entity_set)
            history_all_rel.append(all_rel_set)
            history_all_type.append(all_type_set)

            conversation_triples = set()
            all_kg_element_set = set()

            for ct in history_conversation_triples[-last_n:]:
                #print(ct, type(ct))
                conversation_triples.update(ct)
            
            for es in history_all_entity[-last_n:]:
                all_kg_element_set.update(es)
            for es in history_all_rel[-last_n:]:
                all_kg_element_set.update(es)
            for es in history_all_type[-last_n:]:
                all_kg_element_set.update(es)


            kbelements_in_current_sparql = get_kb_elements(gold_actions)
            missing_element_list=[]
            for e, kbeletype in kbelements_in_current_sparql:
                if e not in all_kg_element_set:
                    missing_element_list.append((e, kbeletype))
            
            kb_elements_in_current_triples = get_kb_elements_in_current_triples(conversation_triples)
            
            if types:
                type_rel_label_dict = get_type_rel_dict(types, kb_elements_in_current_triples)
            else:
                type_rel_label_dict = {}
            
            turn_data={}
            turn_data['input'] = input 
            turn_data['logical_form_normalized'] = logical_form
            turn_data['gold_actions'] = gold_actions
            turn_data['conversation_triples'] = list(conversation_triples.copy())
            turn_data['missing_element_list'] = missing_element_list
            turn_data['turnID'] = f'{fID}#{i}'
            turn_data['question-type'] = user['question-type']
            turn_data['type_rel_label_dict'] = type_rel_label_dict.copy()
            turn_data['description'] = description
            turn_data['answer'] = answer
            if not args.is_train:
                turn_data['results'] = results
            #input_data.append([input, logical_form, gold_actions, list(conversation_triples.copy()), \
            #    missing_element_list.copy(), f'{fID}#{i}', user['question-type'], type_rel_label_dict.copy(), #description])
            input_data.append(turn_data.copy())

    return input_data


def main(args):
    assert args.dataset in args.file_path, f'Something wrong, file path {args.file_path} sould be of the corresponding dataset {args.dataset} '
    allfiles = []
    for f in open(args.file_path, 'r').readlines():
        allfiles.append(f.rstrip())
    
    print('Loading files for ', args.dataset, len(allfiles))

    corpora = {f'{args.dataset}': allfiles}

    for corpus_type in corpora.keys():
        a_lst = [(f, args) for f in corpora[corpus_type]]
        #filelist = corpora[corpus_type]
        preprocess_file((corpora[corpus_type][0], args))
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for processed_input in tqdm(pool.imap_unordered(preprocess_file, a_lst), total=len(a_lst)):
            #print('processed_input ', processed_input)
            for p in processed_input:
                input_dict = p
                dataset.append(input_dict.copy())
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:d}.jsonl".format(corpus_type, p_ct)
                pt_file = os.path.join(args.save_path, pt_file)
                print(f'Saving shard: {pt_file}')
                with open(pt_file, 'w') as savefile:
                    # save.write('\n'.join(dataset))
                    #savefile.write(json.dumps(dataset,  encoding='utf-8', ensure_ascii=False))
                    for l in dataset:
                        l=json.dumps(l, ensure_ascii=False)
                        savefile.write(str(l)+'\n')
                    p_ct += 1
                    dataset = []
                    gc.collect()

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:d}.jsonl".format(corpus_type, p_ct)
            pt_file = os.path.join(args.save_path, pt_file)
            print(f'Saving shard: {pt_file}')
            with open(pt_file, 'w') as savefile:
                # save.write('\n'.join(dataset))
                # convert this in to jsonlines file
                #savefile.write(json.dumps(dataset,  encoding='utf-8', ensure_ascii=False))
                for l in dataset:
                    l=json.dumps(l, ensure_ascii=False)
                    savefile.write(str(l)+'\n')
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
    parser.add_argument("-file_path", required=True)
    parser.add_argument("-save_path", required=True)
    parser.add_argument("-shard_size", default=10000, type=int)
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=10, type=int)
    parser.add_argument('-last_n', default=1, type=int)
    parser.add_argument('-debug', nargs='?',const=False,default=False)
    parser.add_argument('-nentities', required=True, choices=[NEGOLD, NESTRNEL, NEALLENNEL, JOINTNEL])
    parser.add_argument('-types_to_use', required=True, choices=[TGOLD, TLINKED, TJOINT_LINKED, 'NONE'])
    parser.add_argument("-is_train", type=str2bool, nargs='?', const=True, default=False, required=False)
    
    args = parser.parse_args()
    if args.dataset == 'test':
        assert args.is_train == False
    main(args)

