

def tokenize(text : str):
    """ tokenize funciton """
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace("'", "")
    return text.split()


def text_to_indices(text: str, vocab : dict) -> list:
    """Function to convert words to numerical indices """
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append( vocab[token] )
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text




