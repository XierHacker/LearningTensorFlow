import os
import io
import re
import time
import unicodedata
import numpy as np
import tensorflow as tf

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
    return tensor, lang_tokenizer



def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


if __name__=="__main__":
    # en_sentence = u"May I borrow this book?"
    # sp_sentence = u"¿Puedo tomar prestado este libro?"
    # print(preprocess_sentence(en_sentence))
    # print(preprocess_sentence(sp_sentence).encode('utf-8'))
    #
    # en, sp = create_dataset(path="../corpus/spa.txt", num_examples=None)
    # print(en)
    #print(sp)

    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path="../corpus/spa.txt", num_examples=None)
    print("input_tensor:\n",input_tensor)
    print("target_tensor:\n",target_tensor)

    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    print("max_length_targ:",max_length_targ)
    print("max_length_inp:",max_length_inp)

    print("Input Language; index to word mapping")
    convert(inp_lang, input_tensor[0])
    print()
    print("Target Language; index to word mapping")
    convert(targ_lang, target_tensor[0])

    vocab_inp_size = len(inp_lang.word_index) + 1
    print("vocab_inp_size:",vocab_inp_size)
    vocab_tar_size = len(targ_lang.word_index) + 1
    print("vocab_tar_size:",vocab_tar_size)