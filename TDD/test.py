#pip import pytest
#su linea di comando passo
#i due puntini verdi indicado i test passati ..

from numlib import split_text_num,abs_diff

def test_split_text_num():
    text= "27484   55634"
    assert (27484,55634) == split_text_num(text) #restituisco una tupla

def test_split_text_invalid_num():
    try:
        pair= split_text_num("ciao   12345")
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"Should not have passed. Got:{pair}"
        )

def test_abs_diff():
    a=42
    b=67
    assert abs_diff(a,b)==abs_diff(b,a)==25

