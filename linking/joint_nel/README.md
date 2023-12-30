This is the script for joint linking entities and types.

### Create string automaton for disambiguation
```
Check disam/README.md for instructions
Copy the output automaton.pkl here
```

### Linking
```
python joint_nel.py -data_path path_to_spice_dataset_withsubgraph -save_path output_path/joint_tag_preprocess -split valid -kg_folder path_containing_knowledge_graph_folder
```