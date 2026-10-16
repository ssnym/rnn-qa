from src.config import STRIP_CHARS, UNK_TOKEN

def tokenize(text : str):
    """ tokenize funciton """
    text = text.lower()
    for char in STRIP_CHARS:
        text = text.replace(char, "")
    return text.split()


def text_to_indices(text: str, vocab : dict) -> list:
    """Convert words to vocab indices, using the UNK index for unknown words. """
    unk_index = vocab[UNK_TOKEN]
    return [vocab.get(token, unk_index) for token in tokenize(text)]




