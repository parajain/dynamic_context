import json
import tqdm
import re
import sys, os


def get_value(question):
    if 'min' in question.split():
        value = '0'
    elif 'max' in question.split():
        value = '0'
    elif 'exactly' in question.split():
        value = re.search(r'\d+', question.split('exactly')[1])
        if value:
            value = value.group()
    elif 'approximately' in question.split():
        value = re.search(r'\d+', question.split('approximately')[1])
        if value:
            value = value.group()
    elif 'around' in question.split():
        value = re.search(r'\d+', question.split('around')[1])
        if value:
            value = value.group()
    elif 'atmost' in question.split():
        value = re.search(r'\d+', question.split('atmost')[1])
        if value:
            value = value.group()
    elif 'atleast' in question.split():
        value = re.search(r'\d+', question.split('atleast')[1])
        if value:
            value = value.group()
    else:
        print(f'Could not extract value from question: {question}')
        value = '0'

    return value


def map_format(line):
    outline = {}
    logical_forms_str = ' '.join([a[1] for a in line['logical_forms_str']])
    outline['question'] = line['inputs']
    outline['turnID'] = line['turnID']
    outline['question_type'] = line['question-type']
    pred_str = ' '.join(line['predicted_lf'])
    outline['actions'] = pred_str
    if '[UNK]' in pred_str:
        ques = ' '.join(outline['question'])
        val = get_value(ques)
        if val:
            pred_str = pred_str.replace('[UNK]', val)
            print('replace')
            outline['actions'] = pred_str
    
    outline['sparql_delex'] = logical_forms_str
    outline['is_correct'] = line['string_match']
    outline['description'] = line['description']
    outline['answer'] = line['answer']
    outline['results'] = line['results']
    outline['prev_results'] = 'prev_results'
    return outline


def split_file(input_filename, outdir):
    data = open(input_filename, 'r')
    qtype_data = {}

    for line in data:
        line = json.loads(line)
        line = map_format(line)
        question_type = line['question_type']
        d = qtype_data.get(question_type, [])
        d.append(line)
        qtype_data[question_type] = d

    for qt in qtype_data.keys():
        fn = outdir + '/' +  qt + '.json'
        print('Writing ' + fn)
        fp = open(fn, 'w')
        json.dump(qtype_data[qt], fp, ensure_ascii=False, indent = 2)
    
    return None


def main():
    ifile = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir,exist_ok=True)
    split_file(ifile, outdir)

if __name__ == '__main__':
    main()
