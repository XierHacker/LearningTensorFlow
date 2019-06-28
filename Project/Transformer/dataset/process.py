import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 

#parameters
MAX_LENGTH = 40

def get_dataset():
    dataset,info=tfds.load(name="ted_hrlr_translate/pt_to_en",with_info=True,as_supervised=True)
    #print("dataset:\n",dataset)
    # print("info:\n",info)

    train_dataset=dataset["train"]
    val_dataset=dataset["validation"]

    #load subword and restore tokenizer
    tokenizer_en=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/en")
    tokenizer_pt=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/pt")

    # sample_string = 'Transformer is awesome.'
    # tokenized_string = loaded_tokenizer_en.encode(sample_string)
    # print ('Tokenized string is {}'.format(tokenized_string))

    # for ts in tokenized_string:
    #     print(ts,"---->",loaded_tokenizer_en.decode([ts]))

    # sample_string = 'este é o primeiro livro que eu fiz.'
    # tokenized_string = loaded_tokenizer_pt.encode(sample_string)
    # print ('Tokenized string is {}'.format(tokenized_string))

    # for ts in tokenized_string:
    #     print(ts,"---->",loaded_tokenizer_pt.decode([ts]))

    def encode(lang1,lang2):
        '''
        Add a start and end token to the input and target.
        '''
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size+1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size+1]
        return lang1, lang2

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,tf.size(y) <= max_length)

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])


    train_dataset = train_dataset.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)


    # val_dataset = val_examples.map(tf_encode)
    # val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    return train_dataset



