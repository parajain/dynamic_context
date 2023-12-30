import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()


    ###To merge

    parser.add_argument("--share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--noise", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--pass_silver", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--dec_dropout", default=0.2, type=float)
    parser.add_argument("--dec_layers", default=1, type=int)
    parser.add_argument("--dec_hidden_size", default=768, type=int)
    parser.add_argument("--dec_heads", default=8, type=int)
    parser.add_argument("--dec_ff_size", default=2048, type=int)
    parser.add_argument("--enc_hidden_size", default=512, type=int)
    parser.add_argument("--enc_ff_size", default=512, type=int)
    parser.add_argument("--enc_dropout", default=0.2, type=float)
    parser.add_argument("--enc_layers", default=1, type=int)
    parser.add_argument("--clayers", default=1, type=int)
    parser.add_argument("--take_context_ent_n", default=1, type=int)
    parser.add_argument("--take_current_n", default=0, type=int)




    ####

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # data
    parser.add_argument('--data_path', default='/data/final/csqa')
    parser.add_argument('--train_dataset_size', default=1368649, type=int)
    parser.add_argument('--val_dataset_size', default=149617, type=int)

    # experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--path_error_analysis', default='experiments/error_analysis', type=str)
    parser.add_argument('--path_inference', default='experiments/inference', type=str)
    parser.add_argument('--tbd', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--expname', type=str, required=True)
    parser.add_argument("--use_wandb", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use_wandb.")

    # task
    parser.add_argument('--task', default='multitask', choices=['multitask',
                                                                'logical_form',
                                                                'ner',
                                                                'coref',
                                                                'graph'], type=str)

    # model
    parser.add_argument('--model_type',default='simple', choices=['simple', 'dyn', 'dyn-type', 'que_prior_attn',
                                                                'que_prior'], type=str)
    parser.add_argument('--emb_dim', default=768, type=int)
    parser.add_argument('--dropout', default=0.5, type=int)
    parser.add_argument('--heads', default=6, type=int)
    #parser.add_argument('--tgt_layers', default=2, type=int)
    #parser.add_argument('--tgt_emb', default=200, type=int)

    parser.add_argument('--max_positions', default=1000, type=int)
    parser.add_argument('--pf_dim', default=300, type=int)
    parser.add_argument('--graph_heads', default=2, type=int)
    parser.add_argument('--bert_dim', default=3072, type=int)
    parser.add_argument('--type_acc', default='pooler_output', choices=['pooler_output',
                                                                'attn',
                                                                'full',
                                                                'attn-full'], type=str)
    parser.add_argument('--iter_type', default='binary_batch', choices=['binary_batch',
                                                                'chunked_jsonl',
                                                                'jsonl'], type=str)
    parser.add_argument('--sim', default='dot', choices=['dot', 'cos','attn'], type=str)

    # training
    parser.add_argument('--optim', type=str, required=True)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--start_lr', default=1e-4, type=float)
    parser.add_argument('--end_lr', default=1e-7, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--lambda1', default=0.8, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=4000, type=float)
    parser.add_argument('--factor', default=1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=5, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--real_batch_size', default=-1, type=int, required=False)
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument("--fine_tune_bert", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="fine_tune_bert.")
    parser.add_argument("--learn_loss_weights", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Learn loss interpolation weights.")
    parser.add_argument("--schedule_loss_weight", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Learn loss schedule.")

    # test and inference
    parser.add_argument('--model_path', default='experiments/snapshots/', type=str)
    parser.add_argument('--inference_partition', default='test', choices=['val', 'test'], type=str)
    parser.add_argument('--question_type', default='Clarification',
        choices=['Clarification',
                'Comparative Reasoning (All)',
                'Logical Reasoning (All)',
                'Quantitative Reasoning (All)',
                'Simple Question (Coreferenced)',
                'Simple Question (Direct)',
                'Simple Question (Ellipsis)',
                'Verification (Boolean) (All)',
                'Quantitative Reasoning (Count) (All)',
                'Comparative Reasoning (Count) (All)'], type=str)

    return parser
