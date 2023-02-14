# Generate games

First step is to choose the dimensions you want to consider, and to create saved games using these by playing PyRat games. For example, you can run (from PyRat root directory) a command similar to the following:

```
python3 pyrat.py -p 10 -x 7 -y 5 -md 0 -d 0 --nonsymmetric --rat AIs/manh.py --python AIs/manh.py --tests 1000 --nodrawing --synchronous --save
```

# Generate the dataset

Open the `generate_dataset.py` script and update the `convert_input` function, that should parse the saved games to make something useful out of it to train a model. Then run (from here):

```
python3 generate_dataset.py <path_to_saved_games>
```

# Train a classifier

Open the `train.py` file and update the model you want to use, as well as the training parameters. Then run (from here):

```
python3 train_network.py <path_to_pyrat_dataset>
```

# Test in PyRat

Now, it's time to test if your AI is able to beat an opponent. Open the `supervised_player.py` file, and update the `TRAINED_MODEL_PATH` constant to set the path to the classifier you want to use. The path should be relative to the PyRat root directory.

Then run the following command, changing the needed parameters. Make sure you use the same settings (width/height) as during training, or your AI will crash:

```
python3 pyrat.py -p 10 -x 7 -y 5 -d 0 -md 0 --rat AIs/manh.py --python <path_to_supervised_player.py> --nonsymmetric
```
