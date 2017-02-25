# Recurrent Neural Network Models

This part implements variants of Recurrent Neural Network models, including (bi-)LSTM, (bi-)GRU, (bi-)basic-RNN. 

### Dataset

These models are tested with sentiment analysis task and you can download the [Standford Sentiment Treebank](http://nlp.stanford.edu/sentiment/) dataset. 
To process the data, I employ an open-source python library, [pytreebank](https://github.com/JonathanRaiman/pytreebank). You should install it with `pip` first.

And I use the pre-trained Glove embeddings, downloaded from [here](http://nlp.stanford.edu/projects/glove/).

### Configuration

You can set the data dir and model hyper-parameters at `src/config.ini`. 

### Usage

Train the model:

```
python3 main.py --train
```

Test the model:

```
python3 main.py --test
```
 
