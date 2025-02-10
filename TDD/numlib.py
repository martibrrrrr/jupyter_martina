#qui scrivo le funzioni

def split_text_num(text):
    """Split the given text into two integers"""
    a,b = text.split()
    return(int(a),int(b))
# help(numlib.split_text_num). per leggere l'help che ho scirtto
def abs_diff(a,b):
    """Return the absolute value of the difference
     between the two numbers"""
    return abs(a-b)
