import numpy as np
import re

def sep2char(line):
    charline=""
    if line=="":
        return charline
    for char in line:
        charline+=char+" "
    print("charline:",charline)
    return charline


file_out=open(file="processd.zh",mode="a",encoding="utf-8")

with open("./train.tags.en-zh.zh",encoding="utf-8") as in_file:
    lines=in_file.readlines()
    for line in lines:
        line=line.strip()
        print("line:",line)
        line=re.sub(pattern=" ",repl="",string=line)
        print("line:",line)
        sep2char(line)
        line=sep2char(line)
        file_out.write(line+"\n")

file_out.close()




