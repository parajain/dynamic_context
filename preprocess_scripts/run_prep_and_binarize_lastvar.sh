#!/bin/bash
set -e
last_n=$1
split=$2
bs=$3
is_train=$4
out_path=$5
CODE_DIR=$PWD
echo "Code dir"$CODE_DIR
cd $CODE_DIR
echo "Working dir "${PWD}
file_path=$CODE_DIR"/file_lists_temp/"${split}"list.txt"
echo "File list path: "${file_path}
out_dir_base='processed_data_jointnel_last_'${last_n}

if [ -d "${out_path}/${out_dir_base}/${split}" ]; then
  echo "Directory ${out_path}/${out_dir_base}/${split} exist cleanup to avoid overwrite..."
  exit 0
else
    echo "Creating ${out_path}/${out_dir_base}/${split}"
fi


mkdir -p ${out_path}/${out_dir_base}/${split}
echo "Starting preprocessing ${split} ${split_number}... last: "${last_n}
python preprocess.py -file_path ${file_path} -save_path ${out_path}/${out_dir_base}/${split} -dataset ${split} -last_n ${last_n} -n_cpus 1 -nentities "jointnel" -types_to_use "joint_link" #-debug 1
echo "Done ${split}"


#python ~/notify_msg.py "prep for ${split} done "


cd ${out_path}/${out_dir_base}/${split}
echo "merge ${split} at "${PWD}
cat ${split}.*.jsonl > ../all${split}_${last_n}.jsonl

cd $CODE_DIR
echo "Working dir "${PWD}
echo "Starting binary"


n_cpus=1
data_dir=${out_path}/${out_dir_base}"/"
out_dir=$out_path"/binary_jointnel_mainsplit_bs"$bs"_last"${last_n}"/"

mkdir -p $out_dir

outsplit=$out_dir$split"/"

if [ -d "${outsplit}" ]; then
  echo "Directory ${outsplit} exist..."
  exit 0
else
    echo "Not exist creating ${outsplit}"
fi

mkdir -p $outsplit
echo $outsplit
python binarize_context_graph.py -save_path $outsplit -n_cpus ${n_cpus} -batch_size ${bs} -outname $split -inp_filename $data_dir"all"$split"_"${last_n}".jsonl" -is_train ${is_train} -shard_size 100
echo "Binarize complete!"
#python ~/notify_msg.py "binarize ${split} done"

