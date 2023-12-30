## Scripts for string match based ner and nel
Tested for python version 3.8.13 and pyahocorasick 1.4.4


### Create string automata for disambiguation
```
python create_automaton.py -data_path spice_dataset/ -kg_path kg_path/wikidata_proc_json/wikidata_proc_json_2/
```

### If you are facing encoding issues.
```
python redump_ascii_disamb_list.py
```