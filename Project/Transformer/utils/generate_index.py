'''
generate subword index files in `../index_files/`
'''

import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 

dataset,info=tfds.load(name="ted_hrlr_translate/pt_to_en",with_info=True,as_supervised=True)
print("dataset:\n",dataset)
# print("info:\n",info)

train_dataset=dataset["train"]
val_dataset=dataset["validation"]

#generate english subword files
tokenizer_en=tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_generator=(en.numpy() for pt,en in train_dataset),
    target_vocab_size=2**13
)
tokenizer_en.save_to_file(filename_prefix="../index_files/en")

#generate Portugese subword files
tokenizer_pt=tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_generator=(pt.numpy() for pt,en in train_dataset),
    target_vocab_size=2**13
)
tokenizer_pt.save_to_file(filename_prefix="../index_files/pt")



# loaded_tokenizer_en=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/en")
# sample_string = 'Transformer is awesome.'

# tokenized_string = loaded_tokenizer_en.encode(sample_string)
# print ('Tokenized string is {}'.format(tokenized_string))