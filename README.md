### Installation
The code was tested using 

```
pytorch=1.12.1=py3.7_cuda11.3_cudnn8.3.2_0
Other major dependencies are present in requirements.txt
```

### Linking
Code for entity linking is present in ```linking``` folder

Joint Linking : ```linking/joint_nel/README.md```

Disambiguation: ```linking/disamb/README.md```

Subgraph: ```linking/subgraph_util/README.md```

Wikidata server: ```https://github.com/EdinburghNLP/SPICE/tree/main/sparql-server```


### Preprocessing

Please check: ```preprocess_scripts/README.md```

### Train
Training command: 

```
bash train_command.sh
```

### Inference

```bash
cd infer_scripts
checkpoint='checkpoint_path'
data_path='path_to_preprocessed_binary_files/'
exp_name='expname'

bash inference.sh $data_path $exp_name $checkpoint 
```

- Post process in SPICE evaluation format

```
python postprocess.py output_path/test_res.jsonl output_path/split
```