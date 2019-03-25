<h1> IFT6135 Assignment 2 Practical Solutions </h1>
<h2> Guide to recreating our answers: </h2>
<h3>Question 4:</h3>
<h3>4.1: [DONE]</h3>
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best<br>
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best<br>
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best<br>
<br>
4.2: [DONE]<br>
python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9

4.3: [DONE]
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=15 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=15 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=30 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.3
python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=40 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --num_epochs=50
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=15 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.5
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=20 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.5
python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.2
python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.9
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.8

FIGURES & TABLES: [DONE]
Plot figures of models in 'final_res' directory by running the following commands:
python plot_script.py --Q=4.1
python plot_script.py --Q=4.4
python plot_script.py --Q=4.5

############################
####### QUESTION 5 #########
############################

5.1: [DONE]
First, place the files {RNN_best_params.pt, GRU_best_params.pt, TRANSFORMER_best_params.pt} in the directory 'IFT6135/Assignment2/numpy_files/best_params/'. Then, run the following:
python Q5-ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.1
python Q5-ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.1
python Q5-ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --Q=5.1
python plot_script.py --Q=5.1

5.2: [DONE]
python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.2
python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.2
python plot_script.py --Q=5.2

5.3: [DONE]
python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3
python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3
python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3
python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3

