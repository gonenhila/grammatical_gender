# -*- coding: utf-8 -*-

import codecs
import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from numpy import linalg as LA
import argparse
import json
from numpy import linalg as LA
import operator
from collections import defaultdict
from de_lemmatizer.lemmatizer import LOOKUP
import json
import random
from demorphy import Analyzer
from random import shuffle
import ast
import itertools
random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file")
parser.add_argument("--output", help="output file")
parser.add_argument("--lang", default='de', help="output file")
parser.add_argument("--win", default=4, help="window size (for each size)")
parser.add_argument("--lemmatize", help="type of lemmatization")
parser.add_argument("--plus", action='store_true', help="add manual tags")
parser.add_argument("--rand", action='store_true', help="choose one of the options at random")
parser.add_argument("--hybrid", action='store_true', help="use hybrid dictionary")
parser.add_argument("--first", action='store_true', help="use first possible lemma")
parser.add_argument("--different", action='store_true', help="use different lemma for each instance")


def print_pairs(lang, vocab, to_male_dict=None, to_female_dict=None, to_lemma_dict=None, to_neut_dict=None, to_options_dict=None):
    """
    print pairs into file, required for training word2vecf
    supports the different types of lemmatizations, as discussed in the paper
    """
    print ('printing output ...')
    j = 0
    with codecs.open(args.input, 'r', 'utf-8') as f, codecs.open(args.output, 'w', 'utf-8') as f_out:
        for l in tqdm(f):
            l = l.strip().lower().split()
            for i,w in enumerate(l):
                if w in vocab:
                    start = max(0, i - args.win)
                    end = i + args.win + 1
                    for c in l[start:end]:
                        if w != c and c in vocab:
                            if lang == 'it':
                                if c in to_male_dict and to_male_dict[c] == None or c in to_female_dict and to_female_dict[c] == None:
                                    continue
                                if args.lemmatize == 'basic':
                                    if args.different:
                                        if c in to_options_dict:
                                            opts = to_options_dict[c]
                                            shuffle(opts)
                                            f_out.write(w + ' ' + opts[0] + '\n')
                                        else:
                                            f_out.write(w + ' ' + c + '\n')
                                    else:
                                        f_out.write(w + ' ' + to_lemma_dict.get(c, c) + '\n')
                                elif args.lemmatize == 'to_masc':
                                    f_out.write(w + ' ' + to_male_dict.get(c, c) + '\n')
                                elif args.lemmatize == 'to_fem':
                                    f_out.write(w + ' ' + to_female_dict.get(c, c) + '\n')
                                elif not args.lemmatize:
                                    f_out.write(w + ' ' + c + '\n')

                            if lang == 'de':
                                if args.lemmatize == 'basic':
                                    if args.different:
                                        if c in to_options_dict:
                                            opts = to_options_dict[c]
                                            shuffle(opts)
                                            f_out.write(w + ' ' + opts[0] + '\n')
                                        else:
                                            f_out.write(w + ' ' + c + '\n')
                                    else:
                                        f_out.write(w + ' ' + to_lemma_dict.get(c, c) + '\n')
                                elif args.lemmatize == 'to_masc':
                                    f_out.write(w + ' ' + to_male_dict.get(c, c) + '\n')
                                elif args.lemmatize == 'to_fem':
                                    if c in to_female_dict and to_female_dict[c] == None:
                                        continue
                                    f_out.write(w + ' ' + to_female_dict.get(c, c) + '\n')
                                elif args.lemmatize == 'to_neut':
                                    if c in to_neut_dict and to_neut_dict[c] == None:
                                        continue
                                    f_out.write(w + ' ' + to_neut_dict.get(c, c) + '\n')
                                elif not args.lemmatize:
                                    f_out.write(w + ' ' + c + '\n')

                            if lang == 'en':
                                f_out.write(w + ' ' + c + '\n')

    print ('done')


def extract_features(anlyss):
    atts = [a for a in dir(anlyss) if not a.startswith('__')]
    for att in atts:
        print('att', att)
        print(getattr(anlyss, att))


def extract_lemmas(lang):
    """
    Returns dictionaries of the lemmas and words in language 'lang' (with the respective features)
    """
    if lang == 'it':
        words = defaultdict(list)
        lemmas = defaultdict(list)
        with open('../data/lemmatizer_unique.txt', 'r', encoding='latin-1') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    atts = l[2].split(':')
                    if len(atts) > 1:
                        features = set(atts[1].split('+'))
                    else:
                        features = None
                    pos = set(atts[0].split('-'))
                    words[l[0]].append((l[1], pos, features))
                    lemmas[l[1]].append((l[0], pos, features))

    if lang =='de':
        analyzer = Analyzer(char_subs_allowed=True)

        words = defaultdict(list)
        lemmas = defaultdict(list)
        for w in vocab:
            try:
                s = analyzer.analyze(w)
            except:
                continue
            else:
                if len(s) == 0:
                    continue
                for anlyss in s:
                    features = ast.literal_eval(str(anlyss))
                    words[w].append((features['LEMMA'], features))
                    lemmas[features['LEMMA']].append((w, features))

    return words, lemmas


def create_vocab():
    i = 0
    freq = defaultdict(int)
    with open(args.input, 'r', encoding="utf-8") as f:
        for l in tqdm(f):
            for w in l.strip().lower().split():
                freq[w] += 1

    vocab = {}
    for w in freq:
        if freq[w] > 50:
            vocab[w] = freq[w]

    w2i = {w: i for i, w in enumerate(vocab)}

    return vocab, w2i


def return_oppos_gender(word, words, lemmas, to_gender, from_gender, lang):
    """
    Returns a word with the opposite gender (with same other features)
    """

    options = []

    if lang == 'it':

        for (lemma, pos, feat) in words[word]:
            if feat and from_gender in feat:
                #this is from-gender according to feat
                for (w_new, pos_new, feat_new) in lemmas[lemma]:
                    if pos == pos_new and feat.union({to_gender}).difference({from_gender}) == feat_new:
                        options.append(w_new)
            if len(pos) > 1 and from_gender.upper() in pos:
                #this is from-gender according to pos
                for (w_new, pos_new, feat_new) in lemmas[lemma]:
                    if feat == feat_new and pos.union({to_gender.upper()}).difference({from_gender.upper()}) == pos_new:
                        options.append(w_new)
        if word in manual_mapping_gender:
            options = manual_mapping_gender[word]

    if lang == 'de':

        for lemma, features in words[word]:
            if 'GENDER' in features and features['GENDER'] in from_gender:
                for w_new, features_new in lemmas[lemma]:

                    if 'GENDER' in features_new and features_new['GENDER'] == to_gender \
                                                    and len(features_new) == len(features):

                        valid = []
                        for attr in ['CATEGORY', 'NUMERUS', 'PERSON', 'PTB_TAG', 'TENSE']:
                            if attr in features:
                                if attr not in features_new:
                                    valid.append(False)
                                elif features[attr] != features_new[attr]:
                                    valid.append(False)
                        if False not in valid:
                            options.append(w_new)

    if len(options) == 0:
        return None
    if len(options) == 1:
        return options[0]
    if word in options:
        return word

    options_common = list(set([opt for opt in options if opt in vocab]))
    if len(options_common) == 0 and word in vocab:
        return options[0]

    options = options_common

    # If need to chose at random - do so
    if args.rand:
        shuffle(options)
        return options[0]

    # else: choose according to frequency
    freqs = {}
    for opt in options:
        freqs[vocab[opt]] = opt

    # return the option with the closest freq to word
    return freqs[min(freqs, key=lambda x:abs(x-vocab[word]))]


def is_gendered(word, words, from_gender, lang):
    """
    Checks if a given word is gendered or not
    """
    if lang == 'it':
        for (lemma, pos, feat) in words[word]:
            if 'NOUN' in pos:
                return False
            if feat and from_gender in feat:
                return True
            if len(pos) > 1 and from_gender.upper() in pos:
                return True

    if lang =='de':
        for lemma, features in words[word]:
            if features['CATEGORY'] == 'NN':
                return False
            if 'GENDER' in features and features['GENDER'] in from_gender:
                return True

    return False


def extract_all_genders(word, words):

    genders = []
    for lemma, features in words[word]:
        if features['CATEGORY'] == 'NN':
            return []
        if 'GENDER' in features and features['GENDER'] not in genders and features['GENDER'] in ['fem', 'masc', 'neut']:
            genders.append(features['GENDER'])
    return list(set(genders))


#############################################################################################################
#### functions for creating the different kinds of dictionaries, for the different lemmatization methods ####
#############################################################################################################

def create_gender_dict(words, lemmas, to_gender, lang):
    gender_dict = {}

    if lang == 'it':

        if to_gender == 'f':
            from_gender = 'm'
        elif to_gender == 'm':
            from_gender = 'f'
        else:
            raise ValueError('gender is not valid')
    if lang == 'de':
        if to_gender not in ['fem', 'masc', 'neut']:
            raise ValueError('gender is not valid')
        from_gender = ['masc', 'fem', 'neut']
        from_gender.remove(to_gender)

    for w in tqdm(words):
        if w in vocab and is_gendered(w, words, from_gender, lang):
            value = return_oppos_gender(w, words, lemmas, to_gender, from_gender, lang)
            if value:
                gender_dict[w] = value
            else:
                gender_dict[w] = None
    return gender_dict


def create_hybrid_dict(words, to_gender):
    gender_dict = {}

    if to_gender not in ['fem', 'masc', 'neut']:
        raise ValueError('gender is not valid')

    for w in tqdm(words):
        if w in vocab:
            genders = extract_all_genders(w, words)
            if len(genders) == 0:
                continue
            if len(genders) == 1 and to_gender in genders:
                continue
            if len(genders) > 1:
                if to_gender in genders:
                    #take lemma
                    gender_dict[w] = to_lemma_dict[w]
                else:
                    if to_gender == 'masc':
                        to_gender_dict = to_male_dict
                    elif to_gender == 'fem':
                        to_gender_dict = to_female_dict
                    elif to_gender == 'neut':
                        to_gender_dict = to_neut_dict
                    gender_dict[w] = to_gender_dict[w]

    return gender_dict


def create_lemma_dict(words, lang):
    lemma_dict = {}

    for w in tqdm(words):
        if w in vocab:
            options = []
            if lang == 'it':
                for (lemma, pos, feat) in words[w]:
                    options.append(lemma)
            if lang == 'de':
                for lemma, features in words[w]:
                    options.append(lemma)
            if len(options) > 0:
                if args.first:
                    options = sorted(options)
                else:
                    shuffle(options)
                value = options[0]
                lemma_dict[w] = value
            else:
                lemma_dict[w] = None
        if lang =='it' and args.plus:
            if w in manual_mapping_lemma:
                lemma_dict[w] = manual_mapping_lemma[w]

    return lemma_dict


def create_options_dict(words, lang):
    lemma_dict = {}

    count = 0
    for w in tqdm(words):
        if w in vocab:
            options = []
            if lang == 'it':
                for (lemma, pos, feat) in words[w]:
                    options.append(lemma)
            if lang == 'de':
                for lemma, features in words[w]:
                    options.append(lemma)
            if len(set(options)) > 0:
                lemma_dict[w] = list(set(options))
        if lang =='it' and args.plus:
            if w in manual_mapping_lemma:
                lemma_dict[w] = [manual_mapping_lemma[w]]

    return lemma_dict



if __name__ == "__main__":


    args = parser.parse_args()

    with open('../data/mappings/manual_mapping_gender.json', 'r') as datafile:
        manual_mapping_gender = json.load(datafile)

    with open('../data/mappings/manual_mapping_lemma.json', 'r') as datafile:
        manual_mapping_lemma = json.load(datafile)

    vocab, w2i = create_vocab()


    if args.lang == 'it':
        words, lemmas = extract_lemmas(args.lang)
        to_male_dict = create_gender_dict(words, lemmas, 'm', args.lang)
        to_female_dict = create_gender_dict(words, lemmas, 'f', args.lang)
        to_lemma_dict = create_lemma_dict(words, args.lang)

        to_options_dict = create_options_dict(words, args.lang)

        print_pairs(args.lang, vocab, to_male_dict, to_female_dict, to_lemma_dict, to_options_dict=to_options_dict)

    if args.lang == 'de':

        words, lemmas = extract_lemmas(args.lang)
        to_male_dict = create_gender_dict(words, lemmas, 'masc', args.lang)
        to_female_dict = create_gender_dict(words, lemmas, 'fem', args.lang)
        to_neut_dict = create_gender_dict(words, lemmas, 'neut', args.lang)
        to_lemma_dict = create_lemma_dict(words, args.lang)

        to_options_dict = create_options_dict(words, args.lang)


        to_male_dict_hybrid = create_hybrid_dict(words, 'masc')
        to_female_dict_hybrid = create_hybrid_dict(words, 'fem')
        to_neut_dict_hybrid = create_hybrid_dict(words, 'neut')

        if args.hybrid:
            print_pairs(args.lang, vocab, to_male_dict_hybrid, to_female_dict_hybrid, to_lemma_dict, to_neut_dict_hybrid)
        else:
            print_pairs(args.lang, vocab, to_male_dict, to_female_dict, to_lemma_dict, to_neut_dict, to_options_dict=to_options_dict)


    if args.lang == 'en':
        print_pairs(args.lang, vocab)
