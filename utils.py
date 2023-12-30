from __future__ import division
import os, shutil
import re
import time
import json
from turtle import forward
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from arguments import get_parser
#from unidecode import unidecode
from collections import OrderedDict
from transformers import BertTokenizer
#from elasticsearch import Elasticsearch

# import constants
from constants import *

# set logger
logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class Scorer(object):
    """Scorer class"""
    def __init__(self):
        self.tasks = [TOTAL, LOGICAL_FORM, NER, COREF, GRAPH]
        self.results = {
            OVERALL: {task:AccuracyMeter() for task in self.tasks},
            CLARIFICATION: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE: {task:AccuracyMeter() for task in self.tasks},
            LOGICAL: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_COREFERENCED: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_DIRECT: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_ELLIPSIS: {task:AccuracyMeter() for task in self.tasks},
            # -------------------------------------------
            VERIFICATION: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
        }
        self.data_dict = []

    def data_score(self, data, helper, predictor):
        """Score complete list of data"""
        for i, (example, q_type)  in enumerate(zip(data, helper['question_type'])):
            # prepare references
            ref_lf = [t.lower() for t in example.logical_form]
            ref_ner = example.ner
            ref_coref = example.coref
            ref_graph = example.graph

            # get model hypothesis
            hypothesis = predictor.predict(example.input)

            # check correctness
            correct_lf = 1 if ref_lf == hypothesis[LOGICAL_FORM] else 0
            correct_ner = 1 if ref_ner == hypothesis[NER] else 0
            correct_coref = 1 if ref_coref == hypothesis[COREF] else 0
            correct_graph = 1 if ref_graph == hypothesis[GRAPH] else 0

            # save results
            gold = 1
            res = 1 if correct_lf and correct_ner and correct_coref and correct_graph else 0
            # Question type
            self.results[q_type][TOTAL].update(gold, res)
            self.results[q_type][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[q_type][NER].update(ref_ner, hypothesis[NER])
            self.results[q_type][COREF].update(ref_coref, hypothesis[COREF])
            self.results[q_type][GRAPH].update(ref_graph, hypothesis[GRAPH])
            # Overall
            self.results[OVERALL][TOTAL].update(gold, res)
            self.results[OVERALL][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[OVERALL][NER].update(ref_ner, hypothesis[NER])
            self.results[OVERALL][COREF].update(ref_coref, hypothesis[COREF])
            self.results[OVERALL][GRAPH].update(ref_graph, hypothesis[GRAPH])

            # save data
            self.data_dict.append({
                INPUT: example.input,
                LOGICAL_FORM: hypothesis[LOGICAL_FORM],
                f'{LOGICAL_FORM}_gold': example.logical_form,
                NER: hypothesis[NER],
                f'{NER}_gold': example.ner,
                COREF: hypothesis[COREF],
                f'{COREF}_gold': example.coref,
                GRAPH: hypothesis[GRAPH],
                f'{GRAPH}_gold': example.graph,
                # ------------------------------------
                f'{LOGICAL_FORM}_correct': correct_lf,
                f'{NER}_correct': correct_ner,
                f'{COREF}_correct': correct_coref,
                f'{GRAPH}_correct': correct_graph,
                IS_CORRECT: res,
                QUESTION_TYPE: q_type
            })

            if (i+1) % 500 == 0:
                logger.info(f'* {OVERALL} Data Results {i+1}:')
                for task, task_result in self.results[OVERALL].items():
                    logger.info(f'\t\t{task}: {task_result.accuracy:.4f}')

    def write_results(self):
        save_dict = json.dumps(self.data_dict, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{args.path_error_analysis}/error_analysis.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.instances = 0

class Inference(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED)
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}]) # connect to elastic search server
        self.es = Elasticsearch([{'host': 'localhost'}]) # connect to elastic search server
        self.inference_actions = []

    def construct_actions(self, inference_data, predictor, logger):
        logf=open('logf.out', 'a')
        tic = time.perf_counter()
        # based on model outpus create a final logical form to execute
        question_type_inference_data = [data for data in inference_data if args.question_type in data[QUESTION_TYPE]]
        total=len(question_type_inference_data)
        for i, sample in enumerate(question_type_inference_data):
            logf.write(str(i) + '\n')
            logger.info(f"i {i} of {total}")
            predictions = predictor.predict(sample['context_question'])
            actions = []
            logger.info(f"pred complete i {i} of {total}")
            logical_form_prediction = predictions[LOGICAL_FORM]
            ent_count_pos = 0
            for j, action in enumerate(logical_form_prediction):
                if action not in [ENTITY, RELATION, TYPE, VALUE, PREV_ANSWER]:
                    actions.append([ACTION, action])
                elif action == ENTITY:
                    # get predictions
                    context_question = sample[CONTEXT_QUESTION]
                    ner_prediction = predictions[NER]
                    coref_prediction = predictions[COREF]
                    # get their indices
                    ner_indices = OrderedDict({k: tag.split('-')[-1] for k, tag in enumerate(ner_prediction) if tag.startswith(B) or tag.startswith(I)})
                    coref_indices = OrderedDict({k: tag for k, tag in enumerate(coref_prediction) if tag not in ['NA']})
                    # create a ner dictionary with index as key and entity as value
                    ner_idx_ent = self.create_ner_idx_ent_dict(ner_indices, context_question)
                    if str(ent_count_pos) not in list(coref_indices.values()):
                        if args.question_type in [CLARIFICATION, QUANTITATIVE_COUNT] and len(list(coref_indices.values())) == ent_count_pos: # simple constraint for clarification and quantitative count
                            for l, (cidx, ctag) in enumerate(coref_indices.items()):
                                if ctag == str(ent_count_pos-1):
                                    if cidx in ner_idx_ent:
                                        actions.append([ENTITY, ner_idx_ent[cidx][0]])
                                        break
                                    else:
                                        print(f'Coref index {cidx} not in ner entities!')
                                        actions.append([ENTITY, ENTITY])
                                        break
                            try:
                                actions.append([ENTITY, ner_idx_ent.popitem()[1][0]])
                            except:
                                print('No coref indices!')
                                actions.append([ENTITY, ENTITY])
                        elif args.question_type in [VERIFICATION, SIMPLE_DIRECT, CLARIFICATION] and ent_count_pos == 0 and not coref_indices: # simple constraint for verification and simple question (direct)
                            try:
                                actions.append([ENTITY, ner_idx_ent.popitem()[1][0]])
                            except:
                                print('No coref indices!')
                                actions.append([ENTITY, ENTITY])
                        else:
                            # TODO here things get hard, we will need to use all ner entites and see if it works
                            print('No coref indices!')
                            actions.append([ENTITY, ENTITY])
                    else:
                        for l, (cidx, ctag) in enumerate(coref_indices.items()):
                            if ctag == str(ent_count_pos):
                                if cidx in ner_idx_ent:
                                    actions.append([ENTITY, ner_idx_ent[cidx][0]])
                                    break
                                else:
                                    print(f'Coref index {cidx} not in ner entities!')
                                    actions.append([ENTITY, ENTITY])
                                    break
                    # update entity position counter
                    ent_count_pos += 1
                elif action == RELATION:
                    predicate_prediction = predictions[GRAPH]
                    if predicate_prediction[j].startswith('P'):
                        actions.append([RELATION, predicate_prediction[j]])
                    else: # Predicate
                        print(f'Predicate prediction not in correct position: {sample}')
                elif action == TYPE:
                    type_prediction = predictions[GRAPH]
                    if type_prediction[j].startswith('Q'):
                        actions.append([TYPE, type_prediction[j]])
                    else: # Type
                        print(f'Type prediction not in correct position: {sample}')
                elif action == VALUE:
                    try:
                        actions.append([VALUE, self.get_value(sample[QUESTION])])
                    except Exception as ex:
                        print(ex)
                        actions.append([VALUE, '0'])
                elif action == PREV_ANSWER:
                    actions.append([ENTITY, PREV_ANSWER])

            self.inference_actions.append({
                TURN_ID: sample[TURN_ID],
                QUESTION_TYPE: sample[QUESTION_TYPE],
                DESCRIPTION: sample[DESCRIPTION],
                QUESTION: sample[QUESTION],
                ANSWER: sample[ANSWER],
                ACTIONS: actions,
                RESULTS: sample[RESULTS],
                PREV_RESULTS: sample[PREV_RESULTS],
                GOLD_ACTIONS: sample[GOLD_ACTIONS] if GOLD_ACTIONS in sample else [],
                IS_CORRECT: 1 if GOLD_ACTIONS in sample and sample[GOLD_ACTIONS] == actions else 0
            })

            if (i+1) % 100 == 0:
                toc = time.perf_counter()
                print(f'==> Finished action construction {((i+1)/len(question_type_inference_data))*100:.2f}% -- {toc - tic:0.2f}s')

        self.write_inference_actions()

    def create_ner_idx_ent_dict(self, ner_indices, context_question):
        ent_idx = []
        ner_idx_ent = OrderedDict()
        for index, span_type in ner_indices.items():
            if not ent_idx or index-1 == ent_idx[-1][0]:
                ent_idx.append([index, span_type]) # check wether token start with ## then include previous token also from context_question
            else:
                # get ent tokens from input context
                ent_tokens = [context_question[idx] for idx, _ in ent_idx]
                # get string from tokens using tokenizer
                ent_string = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
                # get elastic search results
                es_results = self.elasticsearch_query(ent_string, ent_idx[0][1]) # use type from B tag only
                # add idices to dict
                if es_results:
                    for idx, _ in ent_idx:
                        ner_idx_ent[idx] = es_results
                # clean ent_idx
                ent_idx = [[index, span_type]]
        if ent_idx:
            # get ent tokens from input context
            ent_tokens = [context_question[idx] for idx, _ in ent_idx]
            # get string from tokens using tokenizer
            ent_string = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
            # get elastic search results
            es_results = self.elasticsearch_query(ent_string, ent_idx[0][1])
            # add idices to dict
            if es_results:
                for idx, _ in ent_idx:
                    ner_idx_ent[idx] = es_results
        return ner_idx_ent

    def elasticsearch_query(self, query, filter_type, res_size=50):
        res = self.es.search(index='csqa_wikidata', doc_type='entities', body={'size': res_size, 'query': {'match': {'label': {'query': unidecode(query), 'fuzziness': 'AUTO'}}}})
        results = []
        for hit in res['hits']['hits']: results.append([hit['_source']['id'], hit['_source']['type']])
        filtered_results = [res for res in results if filter_type in res[1]]
        return [res[0] for res in filtered_results] if filtered_results else [res[0] for res in results]

    def get_value(self, question):
        if 'min' in question.split():
            value = '0'
        elif 'max' in question.split():
            value = '0'
        elif 'exactly' in question.split():
            value = re.search(r'\d+', question.split('exactly')[1]).group()
        elif 'approximately' in question.split():
            value = re.search(r'\d+', question.split('approximately')[1]).group()
        elif 'around' in question.split():
            value = re.search(r'\d+', question.split('around')[1]).group()
        elif 'atmost' in question.split():
            value = re.search(r'\d+', question.split('atmost')[1]).group()
        elif 'atleast' in question.split():
            value = re.search(r'\d+', question.split('atleast')[1]).group()
        else:
            print(f'Could not extract value from question: {question}')
            value = '0'

        return value

    def write_inference_actions(self):
        with open(f'{args.path_inference}/{args.model_path.rsplit("/", 1)[-1].rsplit(".", 2)[0]}_{args.inference_partition}_{args.question_type}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(self.inference_actions, indent=4))

def save_checkpoint(state):
    filename = f'{args.snapshots}/{MODEL_NAME}_epoch_{state[EPOCH]}_globalstep_{state[GLOBAL_STEPS]}_v{state[CURR_VAL]:.4f}_{args.task}.pth.tar'
    filename_last = f'{args.snapshots}/{MODEL_NAME}.pth.tar'
    torch.save(state, filename)
    try:
        os.unlink(filename_last)
    except FileNotFoundError:
        pass
    try:
        os.symlink(filename, filename_last)
    except OSError:
        shutil.copy2(filename, filename_last)
    print(filename, filename_last)
    return filename

class OnlyKGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_kg_loss = nn.CrossEntropyLoss()
       
        
    def calculate_kgloss(self, kgout, selected_output_index):
        total_loss = 0
        assert len(selected_output_index) == len(kgout)
        for iteri in range(len(selected_output_index)):
            target = torch.tensor(selected_output_index[iteri])
            l = self.criterion_kg_loss(kgout[iteri], target.to(kgout[iteri].device))
            total_loss =+ l
        return total_loss
    
    def forward(self, kg_output, kg_targets):
        total_kgloss = self.calculate_kgloss(kg_output, kg_targets)
        return total_kgloss

class KGLoss(nn.Module):
    def __init__(self, learn_loss_weights, schedule_loss_weight, ignore_index,  lambda1, model_type):
        super().__init__()
        self.criterion_seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.model_type = model_type
        if 'que' in model_type:
            self.criterion_kg_loss = nn.NLLLoss()
        else:
            self.criterion_kg_loss = nn.CrossEntropyLoss()
        
        self.learn_loss_weights = learn_loss_weights
        self.schedule_loss_weight = schedule_loss_weight
        self.lambda1 = lambda1
        assert not(schedule_loss_weight and learn_loss_weights), f'Both schedule_loss_weight {schedule_loss_weight} and learn_loss_weights {learn_loss_weights} cannot be true simultaneously.'
        if learn_loss_weights:
            print('learn_loss_weights .....')
            self.mml_emp = torch.Tensor([True, True])
            self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))
        if self.schedule_loss_weight:
            self.seq_avg = AverageMeter()
            
    def get_loss_weight(self, seq_avg_loss):
        #return weight for sparql seq loss 
        if seq_avg_loss > 0.1:
            return 0.9
        elif seq_avg_loss > 0.01:
            return 0.8
        elif seq_avg_loss > 0.001:
            return 0.6
        else:
            return 0.5
    
    def calculate_kgloss(self, kgout, selected_output_index):
        total_loss = 0
        assert len(selected_output_index) == len(kgout)
        for iteri in range(len(selected_output_index)):
            target = torch.tensor(selected_output_index[iteri])
            if 'que' in self.model_type:
                l = self.criterion_kg_loss(torch.log(kgout[iteri] + 1e-9 ), target.to(kgout[iteri].device))
            else:
                l = self.criterion_kg_loss(kgout[iteri], target.to(kgout[iteri].device))
            total_loss =+ l
        return total_loss
    
    def forward(self, seq_output, seq_target, kg_output, kg_targets):
        seq_loss = self.criterion_seq_loss(seq_output, seq_target)
        total_kgloss = self.calculate_kgloss(kg_output, kg_targets)
        task_losses = torch.stack((
            seq_loss,
            total_kgloss
        ))

        dtype = task_losses.dtype
        if self.learn_loss_weights:
            stds = (torch.exp(self.log_vars)**(1/2)).to(DEVICE).to(dtype)
            weights = 1 / ((self.mml_emp.to(DEVICE).to(dtype)+1)*(stds**2))
            losses = weights * task_losses + torch.log(stds)
            meanloss = losses.mean()
        if self.schedule_loss_weight:
            #self.seq_avg.update(seq_loss.data)
            self.lambda1 = self.get_loss_weight(seq_loss.data)
            weights = [self.lambda1, 1-self.lambda1]
            meanloss = weights[0] * task_losses[0] + weights[1] * task_losses[1]
        else:
            weights = [self.lambda1, 1-self.lambda1]
            meanloss = weights[0] * task_losses[0] + weights[1] * task_losses[1]
        
        return {
            LOGICAL_FORM: task_losses[0],
            KG_OUT: task_losses[1],
            MULTITASK: meanloss,
            "lambda1": self.lambda1
            #MULTITASK: losses.mean()
        }

def init_weights(model):
    # remove init for  bert params
    # initialize model parameters with Glorot / fan_avg
    for n,p in model.named_parameters():
        if 'bert' in n:
            continue
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    """Linear layer"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    """LSTM layer"""
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m
