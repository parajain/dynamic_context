### Preprocessing data

You will need access to SPICE dataset preprocessed with sub-graphs. That can be accessed at https://github.com/EdinburghNLP/SPICE

Following command runs graph preprocessing that includes context from last_n turns. It also creates binary files with $batch_size batch size that is used during training.

```bash
mkdir processed_data
batch_size=2
last_n=5
is_train=false
bash run_prep_and_binarize_lastvar.sh $batch_size valid $last_n $is_train processed_data
```