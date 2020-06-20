script_path = 'python'
pids_path = 'data/pids.txt'

with open(pids_path, 'r') as f:
    pids = f.read().split('\n')[:-1]

rule all:
    input:
        [f"data/train_input/{pid}.pq" for pid in pids],
        'data/train_input/train.pq',
        'data/train_input/test.pq',
        'data/train_output/eval.tsv'

rule fetch_data:
    output:
        directory("data/clean_tac")
    shell:
        "curl https://archive.ics.uci.edu/ml/machine-learning-databases/00515/data.zip -o data/tac.zip"
        " && unzip data/tac.zip -d data/"

rule format:
    input:
        acc='data/all_accelerometer_data_pids_13.csv',
        tac='data/clean_tac/{pid}_clean_TAC.csv'
    output:
        'data/train_input/{pid}.pq'
    shell:
        'python3 {script_path}/format.py'
        ' --accelerometer-path {input.acc}'
        ' --tac-path {input.tac}'
        ' --output-path {output}'
        ' --pid {wildcards.pid}'

rule downsample:
    input:
        [f'data/train_input/{pid}.pq' for pid in pids]
    output:
        train='data/train_input/train.pq',
        test='data/train_input/test.pq'
    shell:
        'python3 {script_path}/downsample.py'
        ' --train-path {output.train}'
        ' --test-path {output.test}'
        ' --input-paths {input}'

rule train:
    input:
        train='data/train_input/train.pq',
        test='data/train_input/test.pq'
    output:
        'data/train_output/eval.tsv'
    shell:
        'python3 {script_path}/train.py'
        ' --train-path {input.train}'
        ' --test-path {input.test}'
        ' --eval-path {output}'

