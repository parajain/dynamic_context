## Entity Neighborhood Sub-Graphs

Input is a SPICE dataset, this script will extract entity neighborhood sub-graphs for linked entities.  This requires the output of joint_nel.

- Create graph vocab: expansion_vocab.json
```
python precompute_local_subgraphs.py --read_folder input_folder_with_jointlink --write_folder output_folder --json_kg_folder path_containing_knowledge_graph_folder --task vocab 
```
- Add local subgraphs

``` bash
python precompute_local_subgraphs.py –read_folder input_folder_with_jointlink –write_folder output_folder –json_kg_folder path_containing_knowledge_graph_folder –allennlpNER_folder joint_tag_preprocess –partition valid –nel_entities
```
