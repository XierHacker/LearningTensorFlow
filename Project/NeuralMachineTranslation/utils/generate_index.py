import numpy as np
from collections import Counter
import pandas as pd

OUT_FILE="../index_files/en_ids.csv"
IN_FILE_LIST=["../wmt_corpus/processed.en"]
EXTRA_CHARS=["<sos>","<eos>","<unk>"]
VOCAB_SIZE=20000

def generate(file_list,out_file):
    counter=Counter()
    #print("counter:\n",counter)
    for file in file_list:
        with open(file=file,encoding="utf-8",errors="ignore") as in_file:
            lines=in_file.readlines()
            #print("lines:\n",lines)
            for line in lines:
                word_list=line.strip().split(sep=" ")
                #print("word_list:",word_list)
                for word in word_list:
                    counter[word]+=1
    #print("counter:\n",counter)
    #print("counter size:",len(counter))
    most_common=counter.most_common(VOCAB_SIZE)
    #print("most_common:\n",most_common)
    most_common_word=[t[0] for t in most_common]
    #print("most_common_word:\n",most_common_word)

    all_word=EXTRA_CHARS+most_common_word
    print("all_word:", len(all_word))
    ids=[i for i in range(1,len(all_word)+1)]
    # #print("ids:",ids)
    pd.DataFrame(data={"id": ids, "word": all_word}).to_csv(path_or_buf=out_file, index=False, encoding="utf_8")

if __name__=="__main__":
    generate(file_list=IN_FILE_LIST,out_file=OUT_FILE)


