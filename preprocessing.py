import pandas as pd, re, numpy as np


def remove_specials(sentence):
    # elimino tag, # e emoji
    sentence = re.sub(r"(@[A-Za-z0–9_]+)|([^-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)
    # elimino caratteri speciali
    sentence = re.sub(r'\W', ' ', sentence)
    # elimino caratteri rimasti soli dopo eliminazione char speciali
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    # elimino doppi spazi
    sentence = re.sub(r'\s+', ' ', sentence)
    return (sentence)


def remove_urls (sentence):
    sentence = re.sub(r"https?:\ / \ / (www\.)?[-a - zA - Z0–9 @: %._\+~# =]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", ' ', sentence)
    sentence = re.sub(r"[-a - zA - Z0–9 @: %._\+~  # =]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", ' ', sentence)
    sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\-|\/|\?|\=|\&|\%)*\b', '', sentence)
    sentence = re.sub(r'(www)(\w|\.|\-|\/|\?|\=|\&|\%)*\b', '', sentence)
    return sentence





def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    """
         Find substrings that consist of `nchars` non-space characters
         and that are repeated at least `ntimes` consecutive times,
         and replace them with a single occurrence.
         Examples: 
         abbcccddddeeeee -> abcde (nchars = 1, ntimes = 2)
         abbcccddddeeeee -> abbcde (nchars = 1, ntimes = 3)
         abababcccababab -> abcccab (nchars = 2, ntimes = 2)
    """
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)


def substitute_repeats(sentence, ntimes=3):
        # Truncate consecutive repeats of short strings
        for nchars in range(1, 20):
            s = substitute_repeats_fixed_len(sentence, nchars, ntimes)

        return s


def translate_foreign(sentence):
    return sentence


def pre_process(sentence, do_translation):
    sentence = sentence.lower()
    sentence = remove_urls(sentence)
    sentence = remove_specials(sentence)
    sentence = substitute_repeats(sentence)
    if do_translation:
        sentence = translate_foreign(sentence)
        
    return sentence
