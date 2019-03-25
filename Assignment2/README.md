<h1> IFT6135 Assignment 2 Practical Solutions </h1>
<h2> Guide to recreating our results: </h2>
<h3>Question 4</h2>
<h3>4.1: [DONE]</h3>
<ul>
  <li>python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best</li>
  <li>python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best</li>
  <li>python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best</li>
</ul>
<h3>4.2: [DONE]</h3>
<ul>
<li>python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9</li>
<li>python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9</li>
</ul>
<h3>4.3: [DONE]</h3>
<ul>
<li>python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=15 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=15 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35</li>
<li>python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=30 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.3</li>
<li>python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=40 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --num_epochs=50</li>
<li>python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=15 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.5</li>
<li>python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=20 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.5</li>
<li>python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.2</li>
<li>python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.9</li>
<li>python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9</li>
<li>python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.8</li>
</ul>
<h3>FIGURES & TABLES: [DONE]</h3>
Plot figures of models in 'final_res' directory by running the following commands:
<ul>
<li>python plot_script.py --Q=4.1</li>
<li>python plot_script.py --Q=4.4</li>
<li>python plot_script.py --Q=4.5</li>
</ul>
<hr>
<h2>Question 5</h2>
<h3>5.1: [DONE]</h3>
First, place the files {RNN_best_params.pt, GRU_best_params.pt, TRANSFORMER_best_params.pt} in the directory 'IFT6135/Assignment2/numpy_files/best_params/'. Then, run the following:<br><br>
<li>python Q5-ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.1</li>
<li>python Q5-ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.1</li>
<li>python Q5-ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --Q=5.1</li>
<li>python plot_script.py --Q=5.1</li>

<h3>5.2: [DONE]</h3>
<ul>
<li>python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.2</li>
<li>python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.2</li>
<li>python plot_script.py --Q=5.2</li>
</ul>
<h3>5.3: [DONE]</h3>
<ul>
<li>python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3</li>
<li>python Q5-ptb-lm.py --model=RNN --batch_size=20 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3</li>
<li>python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3</li>
<li>python Q5-ptb-lm.py --model=GRU --batch_size=20 --seq_len=70 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --Q=5.3</li>
</ul>
