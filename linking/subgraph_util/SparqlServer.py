from pymantic import sparql
import random


## You can add multiple server instances here
servers = {'servername': 'http://129.215.33.207:9999/blazegraph/namespace/wd/sparq'}


class SparqlServer(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, renew=False, server_name = 'servername'):
        if cls._instance is None or renew:
            if renew:
                cls._instance.s.close()
            print('Creating new instance')
            
            if server_name == 'random':
                r = random.randint(0,2)
                server_name = list(servers.keys())[0]
                print('Server ', server_name)
           
            cls._instance = sparql.SPARQLServer(servers[server_name])
        return cls._instance
