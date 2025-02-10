
#oppure si apre con espressioni regolari
from numlib import split_text_num,abs_diff

with open("input.txt") as fd:
    for idx,line in enumerate(fd):
        a,b=split_text_num(line) #creo il primo test
        diff = abs_diff(a,b)
        print(idx,diff)
