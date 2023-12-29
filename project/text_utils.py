import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def make_tokenizer(titles, max_tokens=1024, sequence_length=128):
    tokenizer = TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=sequence_length
    )
    tokenizer.adapt(titles)
    return tokenizer

#TODO: test
def untokenize(tokenizer, sequence):
    vocab = tokenizer.get_vocabulary()
    return " ".join([vocab[token] for token in sequence])

def tokenize_and_onehot_titles(tokenizer, titles, num_tokens=1024):
    titles = titles.map(tokenizer, num_parallel_calls=tf.data.AUTOTUNE)
    titles = titles.map(lambda x: tf.one_hot(x, num_tokens), num_parallel_calls=tf.data.AUTOTUNE)
    return titles
