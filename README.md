# semant_irl

# Setup

Install NumPy & Tensorflow

Run:
```
./get_data.sh
```

Pythonpath:
```
export PYTHONPATH=$PYTHONPATH:/path/to/semant_irl
```

# Demo

```
python semantirl/demo.py
```

This will print a trajectory in the training set to the terminal

# Experiments:

seq2seq example:
```
python semantirl/models/cmd2act.py
```

This trains a model which maps (initial image, command) -> (list of actions)

