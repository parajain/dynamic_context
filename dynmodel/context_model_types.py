import copy
from multiprocessing import context
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv, GATv2Conv
#from pytorch_transformers import BertModel, BertConfig
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dynmodel.decoder import TransformerDecoder


from constants import *


class TransformerDecoderDynEmbed(TransformerDecoder):

    def forward(self, tgt, memory_bank, state, input_syntax_voc, memory_lengths=None,
                    step=None, cache=None, memory_masks=None, dynEmbed=None):

        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings({0: tgt, 1: dynEmbed, 2: input_syntax_voc})

        assert emb.dim() == 3  # len x batch x embedding_dim

        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input \
                = self.transformer_layers[i](
                output, src_memory_bank,
                src_pad_mask, tgt_pad_mask,
                previous_input=prev_layer_input,
                layer_cache=state.cache["layer_{}".format(i)]
                if state.cache is not None else None,
                step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return output, state

def get_classifier(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


def get_generator(vocab_size, dec_hidden_size, device, input_syntax_voc):
    # gen_func = nn.LogSoftmax(dim=-1)
    # generator = nn.Sequential(
    #     DynamicLinear(dec_hidden_size, vocab_size, input_syntax_voc),
    #     gen_func
    # )
    generator = nn.Sequential(
        DynamicLinear(dec_hidden_size, vocab_size, input_syntax_voc)
    )
    generator.to(device)

    return generator

class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, input_syntax_voc, bias=False, device=None):
        super(DynamicLinear, self).__init__()

        factory_kwargs = {'device': device}
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        self.input_syntax_voc = input_syntax_voc
        if input_syntax_voc:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))


    def forward(self, input):
        input, dynWeights = input[0], input[1]

        #print('generator input', input.shape, (input != input).any())

        scores = []
        for b in range(input.size(0)):
            combinedBias = None
            #if self.bias is not None:
            #    combinedBias = torch.cat((self.bias, self.bias.new(dynWeights.size(1))), dim=0)
            if self.input_syntax_voc:
                combinedEmbeddMatrix = dynWeights[b, :, :]
            else:
                combinedEmbeddMatrix = torch.cat((self.weight, dynWeights[b, :, :]), dim=0)
            scores.append(F.linear(input[b,:], combinedEmbeddMatrix, combinedBias))

        scores = torch.stack(scores)

        return scores

class DynamicEmbedding(nn.Embedding):
    """Just an embedding class that re-writes the forward method. On forward, it will append a dynamic set of weights
    to the embedding class weights."""

    def forward(self, input):
        """Dynamic embed each batch item with dynamic set of vectors"""
        # TODO: if this implementation is too slow, do implement lower level

        input, dynamic, input_syntax_voc = input[0], input[1], input[2]
        assert not input_syntax_voc

        embeddings = []
        #print('input', input.shape)
        for b in range(dynamic.size(0)):
            if input_syntax_voc:
                combinedEmbeddMatrix = dynamic[b, :, :]
            else:
                combinedEmbeddMatrix = torch.cat((self.weight, dynamic[b, :, :]), dim=0)
            tmpEmbeds = F.embedding(
                input[b,:], combinedEmbeddMatrix, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
            embeddings.append(tmpEmbeds)

        embeddings = torch.stack(embeddings)

        #print('*embed*  embeddings', embeddings.shape, (embeddings != embeddings).any())
        return embeddings

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        """attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional) —
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked."""
        #TODO: restore mask argument!!!

        if(self.finetune):
            outputs = self.model(input_ids=x, token_type_ids = segs) #, attention_mask=mask, token_type_ids = segs)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=x, token_type_ids = segs) #, attention_mask=mask, token_type_ids = segs)

        return outputs.last_hidden_state


class CustomBert(nn.Module):
    def __init__(self, large, temp_dir, args):
        super(CustomBert, self).__init__()
        
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.model.config.hidden_size, nhead=12, batch_first=True,dropout=args.enc_dropout), num_layers=args.enc_layers)
        self.finetune = args.fine_tune_bert

    def forward(self, x, segs, mask):
        """attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional) —
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked."""
        #TODO: restore mask argument!!!

        if(self.finetune):
            outputs = self.model(input_ids=x, token_type_ids = segs) #, attention_mask=mask, token_type_ids = segs)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=x, token_type_ids = segs) #, attention_mask=mask, token_type_ids = segs)

        ftlayerout = self.transformer_encoder(outputs.last_hidden_state)
        return ftlayerout

class GATLayer(nn.Module):
    def __init__(self, embed_dim, graph_heads, dropout):
        super(GATLayer, self).__init__()
        self.embed_dim = embed_dim
        self.graph_heads = graph_heads
        self.dropout =dropout
        self.gat = GATv2Conv(self.embed_dim, self.embed_dim, heads=graph_heads, dropout=dropout)
        self.gat_linear_out = nn.Linear((self.embed_dim*graph_heads), self.embed_dim)
    
    def forward(self, x, edge_index):
        graph_out = self.gat(x, edge_index)
        graph_out = self.gat_linear_out(graph_out)
        return graph_out


class GAT(nn.Module):
    def __init__(self, embed_dim, graph_heads, num_layers, dropout):
        super(GAT, self).__init__()
        self.embed_dim = embed_dim
        self.graph_heads = graph_heads
        self.num_layers = num_layers

        def get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.gat_layers = get_clones(GATLayer(embed_dim, graph_heads, dropout), N=self.num_layers)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gat_layers[i](x=x,edge_index =edge_index)
        
        return x



class DynGraphContextModelTypes(nn.Module):
    def __init__(self, args, device, gobal_syntax_vocabulary, checkpoint=None):
        super(DynGraphContextModelTypes, self).__init__()
        self.args = args
        self.device = device
        self.gobal_syntax_vocabulary = gobal_syntax_vocabulary
        global_target_voc_size = len(self.gobal_syntax_vocabulary)
        self.bert = CustomBert(False, None, args)
        self.start_token = gobal_syntax_vocabulary.get_stoi()[BOS_TOKEN]
        self.end_token = gobal_syntax_vocabulary.get_stoi()[EOS_TOKEN]

        # if(args.max_pos>512):
        #     my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
        #     self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        self.global_target_voc_size = global_target_voc_size

        #tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        #if (self.args.share_emb):
        #    tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        tgt_embeddings = DynamicEmbedding(self.global_target_voc_size, self.bert.model.config.hidden_size, padding_idx=0)

        #self.decoder = TransformerDecoder(
        self.decoder = TransformerDecoderDynEmbed(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.global_target_voc_size, self.args.dec_hidden_size, device, False)
        self.generator[0].weight = self.decoder.embeddings.weight
        self.embed_dim = self.bert.model.config.hidden_size
        self.gat = GAT(embed_dim=self.embed_dim, graph_heads=args.graph_heads, num_layers=1, dropout=args.dropout)



        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            #print("Model's state_dict:")
            #for param_tensor in self.state_dict():
            #    if param_tensor=='decoder.embeddings.weight' or param_tensor=='generator.0.weight':
            #        print(param_tensor, "\t", self.state_dict()[param_tensor].sum())
            #    else:
            #        print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        self.to(device)

    
    def mean_pooling(self, model_output, attention_mask):
        #token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def pad_and_encode_logical_form(self, logical_form_token_list, nodelinks, extra_links, gobal_syntax_vocabulary, padded_len_gnodes):
        '''
        IMP: padded_len_gnodes this is used because graph nodes are also padded to len(links) will not count those
        another way is to use torch.Tensor.tolist and graph mask , remove padded node and then attach them in dyn_emb. Which is better but unnecessary compute
        '''
        encoded_logical_form_list = []
        lf_selected_indices, node_lf_mappings = [], []
        assert len(nodelinks) == len(logical_form_token_list) == len(extra_links)
        for lftokens, links, elinks in zip(logical_form_token_list, nodelinks, extra_links):
            lf_vocab_encodings= [gobal_syntax_vocabulary.get_stoi()[BOS_TOKEN]]
            
            #lf_vocab_encodings.extend([self.gobal_syntax_vocabulary.get_stoi()[l] if l in self.gobal_syntax_vocabulary.get_stoi().keys() else self.gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN] for l in lftokens ])
            
            global_vocab_len = len(gobal_syntax_vocabulary)
            for iteridx, (toktype, tok) in enumerate(lftokens):
                if toktype in [ENTITY, RELATION, TYPE]:
                    # lf_sel.append(iteridx+1) # +1 for BOS_TOKEN, BOS_TOKEN is passed into the transformer model
                    # if tok in links:
                    #     nlf_map.append(links.index(tok))
                    # else:
                    #     nlf_map.append(links.index('MISSINGNODE'))
                    #     missing=True
                    #     break
                    if tok in links:
                        lf_vocab_encodings.append(links.index(tok) + global_vocab_len)
                        assert tok not in elinks
                    elif tok in elinks:
                        #lf_vocab_encodings.append(elinks.index(tok) + len(links) +  global_vocab_len)
                        lf_vocab_encodings.append(elinks.index(tok) + padded_len_gnodes +  global_vocab_len)
                    else:
                        #lf_vocab_encodings.append(elinks.index('MISSINGNODE') + len(links) +  global_vocab_len)
                        lf_vocab_encodings.append(elinks.index('MISSINGNODE') + padded_len_gnodes +  global_vocab_len)
                # if token type is an action we will put real action index from vocab
                elif tok in gobal_syntax_vocabulary.get_stoi().keys():
                    lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[tok])
                else:
                    #print(f'{tok} was not in the vocab.')
                    lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[UNK_TOKEN]) # why would this happen ? for digits only?
                
            
            lf_vocab_encodings.append(gobal_syntax_vocabulary.get_stoi()[EOS_TOKEN])
            lf_vocab_encodings = torch.tensor(lf_vocab_encodings, dtype=torch.int64)#, device=self.DEVICE)
            #lf_vocab_encodings = torch.cuda.LongTensor(lf_vocab_encodings)#, device=self.DEVICE)
            encoded_logical_form_list.append(lf_vocab_encodings)
            
        encoded_tensor = pad_sequence(encoded_logical_form_list, batch_first=True, padding_value=gobal_syntax_vocabulary.get_stoi()[PAD_TOKEN])
        return encoded_tensor
    
    
    def graph_encode(self, node_encodings,node_type, edges, cum_num_nodes, device, logger):
        graph_data = []
        combine=False
        if combine:
            raise NotImplementedError('Node type embedding is not added here')
            node_input_ids = torch.tensor(node_encodings['input_ids'], dtype=torch.int64, device=device)
            node_segs = torch.tensor(node_encodings['token_type_ids'], dtype=torch.int64, device=device)
            node_src_attention_mask = torch.tensor(node_encodings['attention_mask'], dtype=torch.uint8, device=device)
            nemb = self.bert(x=node_input_ids, segs=node_segs, mask=node_src_attention_mask)
            embs_list=torch.tensor_split(nemb.pooler_output, cum_num_nodes)
            for iter_idx, e in enumerate(embs_list[:-1]):
                gd = gData(x=e, edge_index=edges[iter_idx].to(device)) 
                graph_data.append(gd)
        else:
            for iter_idx, n in enumerate(node_encodings):

                node_input_ids = n['input_ids'].to(device)
                node_segs = n['token_type_ids'].to(device)
                node_src_attention_mask = n['attention_mask'].to(device)

                try:
                    nemb = self.bert(x=node_input_ids, segs=None, mask=node_src_attention_mask)
                except Exception as e:
                    logger.info(str(e))
                    logger.info('node_input_ids ' + str(node_input_ids.shape))
                    raise e
                #node_type_encodings = torch.tensor(node_type[iter_idx], dtype=torch.int64, device=device)
                #node_type_emb = self.node_type_emb(node_type_encodings)
                mean_pooled_nemb = self.mean_pooling(nemb, node_src_attention_mask)
                #mean_pooled_nemb = torch.cat((mean_pooled_nemb, node_type_emb),dim=1)
                gd = gData(x=mean_pooled_nemb, edge_index=edges[iter_idx].to(device)) 
                graph_data.append(gd)
                
        gdbatch = Batch.from_data_list(graph_data)
        #graph_out = self.gat(gdbatch.x, gdbatch.edge_index)
        #graph_out = self.gat_linear_out(graph_out)
        graph_out = self.gat(gdbatch.x, gdbatch.edge_index)
        # ToDo: fill val 1 ? 
        graph_emb, graph_mask = to_dense_batch(graph_out, gdbatch.batch, fill_value=0) # (tensor, boolean) tensor - bs x max_num_nodes x dim 
        return  graph_emb, graph_mask

  
    def extra_node_encode(self, extra_node_encodings, device, logger):
        extra_node_embeddings = []
        for iter_idx, n in enumerate(extra_node_encodings):
            node_input_ids = n['input_ids'].to(device)
            node_segs = n['token_type_ids'].to(device)
            node_src_attention_mask = n['attention_mask'].to(device)

            try:
                nemb = self.bert(x=node_input_ids, segs=None, mask=node_src_attention_mask)
            except Exception as e:
                logger.info(str(e))
                logger.info('extra input ids ' + str(node_input_ids.shape))
                raise e
            mean_pooled_nemb = self.mean_pooling(nemb, node_src_attention_mask)
            extra_node_embeddings.append(mean_pooled_nemb)

        # stack and fill to make them same size for batch loss cal
        #extra_node_embeddings = torch.stack(extra_node_embeddings, dim=0) 
        extra_node_embeddings = pad_sequence(extra_node_embeddings, batch_first=True, padding_value=0)
        return extra_node_embeddings, None

    def forward(self, batch, logger):
        device = self.generator[0].weight.device
        src_input_dict = batch['input_encodings'] 
        
        # will create attention mask
        input_ids = src_input_dict['input_ids'].to(device=device)
        #segs = src_input_dict['token_type_ids'].to(device=device)
        segs = None
        src_attention_mask = src_input_dict['attention_mask'].to(device=device)
        input_embeddings = self.bert(x=input_ids, segs=segs, mask=src_attention_mask) # [bs, seqlen, 768] is last state
        


        node_type = batch['node_type']
        node_encodings = batch['node_encodings']
        edges = batch['edges']
        graph_emb, graph_mask = self.graph_encode(node_encodings=node_encodings, node_type=node_type, edges=edges, cum_num_nodes=None, device=device, logger=logger)
        #print('graph_emb ', graph_emb.shape)
        
        extra_node_embeddings, ex_mask = self.extra_node_encode(extra_node_encodings=batch['extra_node_encodings'], device=device, logger=logger)
        #print('extra_node_embeddings ', extra_node_embeddings.shape)
        
        dy_emb = torch.cat((graph_emb, extra_node_embeddings), dim=1) # order matters this should be same as order used in pad_and_encode_logical_form
        #print('dy_emb ', dy_emb.shape)
        padded_len_gnodes=graph_emb.size(1)

        tgt = self.pad_and_encode_logical_form(logical_form_token_list=batch['logical_forms_str'], nodelinks=batch['nodelinks'], extra_links=batch['extra_and_type_links'], gobal_syntax_vocabulary=self.gobal_syntax_vocabulary, padded_len_gnodes=padded_len_gnodes)
        tgt = tgt.to(device)
        
        dec_state = self.decoder.init_decoder_state(input_ids, input_embeddings)
        decoder_outputs, state = self.decoder(tgt[:, :-1], input_embeddings, dec_state, False, memory_masks=None, dynEmbed=dy_emb)
        decoder_outputs = self.generator({0: decoder_outputs, 1: dy_emb})

        return decoder_outputs, tgt, padded_len_gnodes


    def _tile(self, x, count, dim=0):
        """
        Tiles x on dimension dim count times.
        """
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

    def predict_greedy(self, batch, logger):
        max_length=500
        with_cache=True        
        
        device = self.generator[0].weight.device
        src_input_dict = batch['input_encodings'] 
        input_ids = src_input_dict['input_ids'].to(device=device)
        batch_size = input_ids.shape[0]
        #segs = src_input_dict['token_type_ids'].to(device=device)
        segs = None # use segs ?
        src_attention_mask = src_input_dict['attention_mask'].to(device=device)
        input_embeddings = self.bert(x=input_ids, segs=segs, mask=src_attention_mask) # [bs, seqlen, 768] is last state
        
        node_type = batch['node_type']
        node_encodings = batch['node_encodings']
        edges = batch['edges']
        graph_emb, graph_mask = self.graph_encode(node_encodings=node_encodings, node_type=node_type, edges=edges, cum_num_nodes=None, device=device, logger=logger)
        padded_len_gnodes=graph_emb.size(1)
        extra_node_embeddings, ex_mask = self.extra_node_encode(extra_node_encodings=batch['extra_node_encodings'], device=device, logger=logger)
        
        dy_emb = torch.cat((graph_emb, extra_node_embeddings), dim=1) # order matters this should be same as order used in pad_and_encode_logical_form

        tgt = self.pad_and_encode_logical_form(logical_form_token_list=batch['logical_forms_str'], nodelinks=batch['nodelinks'], extra_links=batch['extra_and_type_links'], gobal_syntax_vocabulary=self.gobal_syntax_vocabulary, padded_len_gnodes=padded_len_gnodes)
        tgt = tgt.to(device)
        
        
        dec_states = self.decoder.init_decoder_state(input_ids, input_embeddings, with_cache=True)
        
        decoder_input = torch.full([batch_size, 1], self.start_token, dtype=torch.long, device=device)
        results = {}
        results['batch'] = batch
        # track eos separately to break  this for batch > 1
        predictions = []
        for dstep in range(max_length):
            #decoder_input1 = torch.ones(1, 1).fill_(self.start_token).type_as(torch.long).to(device=device)
            dec_out, dec_states = self.decoder(decoder_input, input_embeddings, dec_states,
                                                        False,
                                                        dynEmbed=dy_emb,memory_masks=None,
                                                        step=dstep)
            gi = dec_out[:, -1, :]
            log_probs = self.generator.forward({0: gi, 1: dy_emb})
            _, next_word = torch.max(log_probs, dim = 1)
            predictions.append(next_word)
            #next_word = next_word.data[0] # this is for batch size 1
            #if next_word.item() == self.end_token:
            #   break
            #nip = torch.full([1, 1], next_word, dtype=torch.long, device=device)
            nip = next_word.unsqueeze(dim=1)
            if not with_cache:
                decoder_input = torch.cat([decoder_input,  nip], dim=1)
            else:
                decoder_input = nip
        
        preds = torch.stack(predictions, dim = 1)
        results['predictions'] = [preds]
        return results, padded_len_gnodes
    


    def predict_greedy2(self, batch, logger):
            max_length=500

            with_cache=True

            device = self.generator[0].weight.device
            src_input_dict = batch['input_encodings']
            input_ids = src_input_dict['input_ids'].to(device=device)

            batch_size = input_ids.shape[0]
            #segs = src_input_dict['token_type_ids'].to(device=device)
            segs = None # use segs ?
            src_attention_mask = src_input_dict['attention_mask'].to(device=device)
            input_embeddings = self.bert(x=input_ids, segs=segs, mask=src_attention_mask) # [bs, seqlen, 768] is last state
            #tgt = self.pad_and_encode_logical_form(logical_form_token_list=batch['logical_forms_str'], nodelinks=batch['nodelinks'], gobal_syntax_vocabulary=self.gobal_syntax_vocabulary)
            #tgt = tgt.to(device)


            node_type = batch['node_type']
            node_encodings = batch['node_encodings']
            edges = batch['edges']
            graph_emb, graph_mask = self.graph_encode(node_encodings=node_encodings, node_type=node_type, edges=edges, cum_num_nodes=None, device=device, logger=logger)
            dyn_target_embedds = graph_emb


            dec_states = self.decoder.init_decoder_state(input_ids, input_embeddings.last_hidden_state, with_cache=True)

            decoder_input = torch.full([batch_size, 1], self.start_token, dtype=torch.long, device=device)
            results = {}
            results['batch'] = batch
            # track eos separately to break  this for batch > 1
            predictions = []
            for dstep in range(max_length):
                #decoder_input1 = torch.ones(1, 1).fill_(self.start_token).type_as(torch.long).to(device=device)
                dec_out, dec_states = self.decoder(decoder_input, input_embeddings.last_hidden_state, dec_states,
                                                            False,
                                                            dynEmbed=graph_emb,memory_masks=None,
                                                            step=dstep)
                gi = dec_out[:, -1, :]
                log_probs = self.generator.forward({0: gi, 1: graph_emb})
                _, next_word = torch.max(log_probs, dim = 1)
                predictions.append(next_word)
                #next_word = next_word.data[0] # this is for batch size 1
                #if next_word.item() == self.end_token:
                #   break
                #nip = torch.full([1, 1], next_word, dtype=torch.long, device=device)
                nip = next_word.unsqueeze(dim=1)
                if not with_cache:
                    decoder_input = torch.cat([decoder_input,  nip], dim=1)
                else:
                    decoder_input = nip

            preds = torch.stack(predictions, dim = 1)
            results['predictions'] = preds
            return results


    
    def predict_beam_search(self, batch, logger):
        beam_size = 3
        alpha = 0.95
        batch_size = input_ids.shape[0]
        max_length=200
        min_length = 10
        device = self.generator[0].weight.device
        src_input_dict = batch['input_encodings'] 
        input_ids = src_input_dict['input_ids'].to(device=device)
        #segs = src_input_dict['token_type_ids'].to(device=device)
        segs = None # use segs ?
        src_attention_mask = src_input_dict['attention_mask'].to(device=device)
        input_embeddings = self.bert(x=input_ids, segs=segs, mask=src_attention_mask) # [bs, seqlen, 768] is last state
        tgt = self.pad_and_encode_logical_form(logical_form_token_list=batch['logical_forms_str'], nodelinks=batch['nodelinks'], gobal_syntax_vocabulary=self.gobal_syntax_vocabulary)
        tgt = tgt.to(device)


        node_type = batch['node_type']
        node_encodings = batch['node_encodings']
        edges = batch['edges']
        graph_emb, graph_mask = self.graph_encode(node_encodings=node_encodings, node_type=node_type, edges=edges, cum_num_nodes=None, device=device, logger=logger)
        dyn_target_embedds = graph_emb
        
        dec_states = self.decoder.init_decoder_state(input_ids, input_embeddings, with_cache=True)

        dec_states.map_batch_fn(lambda state, dim: self._tile(state, beam_size, dim=dim))

        
        pooled_src_features = self._tile(input_embeddings, beam_size, dim=0)
        #context_mask = tile(context_mask, beam_size, dim=0)
        dyn_target_embedds = self._tile(dyn_target_embedds, beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        #print('FEEDING TGT', tgt)
        #max_length = tgt.size(1)
        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)
            #decoder_input = tile(tgt[:, step], beam_size, dim=0).view(1, -1)
            #print(decoder_input)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)

            #print('alive_seq', alive_seq)
            #print('decoder_input', step, decoder_input)
            #print('pooled_src_features', step, pooled_src_features.shape)
            #print('dyn_target_embedds', step, dyn_target_embedds)

            #print('Step:',step)
            dec_out, dec_states = self.decoder(decoder_input, pooled_src_features, dec_states,
                                                     False,
                                                     dynEmbed=dyn_target_embedds,memory_masks=None,
                                                     step=step)


            # Generator forward.
            log_probs = self.generator.forward({0: dec_out.transpose(0,1).squeeze(0), 1: dyn_target_embedds})
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids // vocab_size #topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    #finished_hyp = is_finished[i].nonzero().view(-1)
                    finished_hyp = torch.nonzero(is_finished[i], as_tuple=False).view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                #non_finished = end_condition.eq(0).nonzero().view(-1)
                non_finished = torch.nonzero(end_condition.eq(0), as_tuple=False).view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            #src_features = src_features.index_select(0, select_indices)
            pooled_src_features = pooled_src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))


        
        tgt = self.pad_and_encode_logical_form(logical_form_token_list=batch['logical_forms_str'], nodelinks=batch['nodelinks'], gobal_syntax_vocabulary=self.gobal_syntax_vocabulary)
        tgt = tgt.to(device)
        return results, tgt
