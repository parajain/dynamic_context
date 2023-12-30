from glob import glob
from os.path import exists

# spice_data_path is the processed spice dataset path
spice_data_path = 'spice_data'
p = f'{spice_data_path}/valid/*'
outfile='file_lists/validlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

p = f'{spice_data_path}/test/*'
outfile='file_lists/testlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()

p = f'{spice_data_path}/train/*'
outfile='file_lists/trainlist.txt'
assert not exists(outfile)


fp=open(outfile, 'w')

files = glob(p + '/*.json')

for f in files:
    fp.write(f + '\n')

fp.close()
