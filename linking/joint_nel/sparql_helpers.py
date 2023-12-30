import requests
import json
import sys
import os
import logging, traceback
import random

servers = {'rockall': 'http://129.215.33.197:9999/blazegraph/namespace/wd/sparq',
    'bravas': 'http://129.215.33.62:9999/blazegraph/namespace/wd/sparq',
    'kinloch': 'http://129.215.33.82:9999/blazegraph/namespace/wd/sparq',
    'tomo': 'http://129.215.33.207:9999/blazegraph/namespace/wd/sparq'}



class SparqlHelper:
    def __init__(self, load_cache=False,server_name='tomo'):
        self.cache = {}
        self.cache_file = 'cache/ent_type_2hop_subgraph.cache'
        #self.cache_file = 'ent_1hop_type.cache'
        logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('itest.log', 'w'),
                            logging.StreamHandler()
                        ])
        self.logger = logging.getLogger(__name__)
        if server_name == 'random':
            r = random.randint(0,2)
            self.server_name = list(servers.keys())[r]
            print('Server ', self.server_name)
        else:
            self.server_name = server_name
        
        if load_cache and os.path.exists(self.cache_file):
            print(f'Loading ent cache from: {self.cache_file}')
            self.cache = json.load(open(self.cache_file, 'r'))

    def get_2hop_types_triples(self, ent):
        server_name = self.server_name
        # get all variables and convert the output to triples for graph connection
        query = 'select ?r1 ?n1 ?t1 ?r2 ?n2 ?t2 where { wd:' +ent+ ' ?r1 ?n1 . ?n1 wdt:P31 ?t1 . OPTIONAL {?n1 ?r2 ?n2 . ?n2 wdt:P31 ?t2} }'
        try:
            #ent in self.cache.keys():
            results = self.cache[ent]
        except:
            try:
                results = self.run_q(query, servers[server_name])
                self.cache[ent] = results
            except Exception as e:
                print(traceback.format_exc())
                self.logger.info(f'Failed for query  {query}')
                raise e
        
        type_triples, type_list = self.get_types_and_triples(results, ent)

        return type_list, type_triples

    def get_2hop_only_types(self, ent, server_name='kinloch'):
        if ent in self.cache.keys():
            return self.cache[ent]
        
        # get all variables and convert the output to triples for graph connection
        query = 'select distinct ?t1 ?t2 where { wd:'+ent+' ?r1 ?n1 . ?n1 wdt:P31 ?t1 OPTIONAL {?n1 ?r2 ?n2 . ?n2 wdt:P31 ?t2} }'
        varBindings = self.get_results(self.run_q(query, servers[server_name]))
        type_list = set()
        for k in varBindings.keys():
            type_list.update(varBindings[k])
        self.cache[ent] = list(type_list)
        return list(type_list)
    
    def get_1hop_only_types(self, ent, server_name='kinloch'):

        #if ent in self.cache.keys():
        #    return self.cache[ent]
        
        # get all variables and convert the output to triples for graph connection
        query = 'select distinct ?t1 where { wd:'+ent+' ?r1 ?n1 . ?n1 wdt:P31 ?t1  }'
        varBindings = self.get_results(self.run_q(query, servers[server_name]))
        type_list = set()
        for k in varBindings.keys():
            type_list.update(varBindings[k])
        self.cache[ent] = list(type_list)
        raise NotImplementedError('Cache has to be different, todo fix')
        return list(type_list)

    def save_cache(self):
        if os.path.exists(self.cache_file):
            print(f'Loading old ent cache from: {self.cache_file}')
            old_cache = json.load(open(self.cache_file, 'r'))
            print('Updating cache')
            self.cache.update(old_cache)
        print('Saving cache.. ')
        json.dump(self.cache, open(self.cache_file, 'w'), indent=2)
        print('Saved.')
        
    
    def run_q(self, query,link):
        acceptable_format = 'application/sparql-results+json'
        headers = {'Accept': acceptable_format}
        response = requests.post(link ,data={'query': query}, headers=headers)
        t = response.content
        outjson = json.loads(t)
        return outjson
        

    def get_results(self, results):
        if 'boolean' in results.keys():
            print(results['boolean'])
            return results['boolean'], 'boolean'
        else:
            varBindings = {}
            #assert len(results['head']['vars']) == 1
            for var in results['head']['vars']:
                varBindings[var] = []
                for bin in results['results']['bindings']:
                    if var in bin.keys():
                        varBindings[var].append(bin[var]['value'].split('/')[-1])
            return varBindings

    def get_types_and_triples(self, results, ent):
        '''
        for each row of results:
    	triples = [ent, r1, n1] , [n1, P31, t1] , [n1, r2, n2], [n2, P31, t2]

        Q: how to do this after filtering types ? 
        - create list based on types: 
            t1 - ([ent, r1, n1] , [n1, P31, t1]), ()
            t2 - ([n1, r2, n2], [n2, P31, t2]), ()

        there could be multiple triples per type based on base entity. How do u choose ? All?
        '''
        REL1 = 'r1'
        REL2 = 'r2'
        N1 = "n1"
        N2 = "n2"
        P31 = 'P31'
        T1= "t1"
        T2= "t2"

        type_triples = {}
        type_list = set()
        vars = results['head']['vars']

        def add_triple(t, triples):
            if t in type_triples.keys():
                for triple in triples:
                    type_triples[t].append(triple)
            else:
                type_triples[t] = []
                for triple in triples:
                    type_triples[t].append(triple)
        
        # assert vars in []
        for bin in results['results']['bindings']:
            node1 = bin[N1]['value'].split('/')[-1]
            rel1 = bin[REL1]['value'].split('/')[-1]
            type1 = bin[T1]['value'].split('/')[-1]
            type_list.add(type1)

            add_triple(type1, [(ent, rel1, node1) , (node1, P31, type1) ])
            if T2 in bin.keys():
                node2 = bin[N2]['value'].split('/')[-1]
                rel2 = bin[REL2]['value'].split('/')[-1]
                type2 = bin[T2]['value'].split('/')[-1]
                add_triple(type2, [(node1, rel2, node2) , (node2, P31, type2)])
                type_list.add(type2)
        
        return type_triples, type_list


