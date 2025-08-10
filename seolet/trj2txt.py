import math

import numpy as np
from geoletrld.utils import Trajectories
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from tqdm.auto import tqdm

from seolet.resampling import time_resample, travel_distance_resample, absolute_distance_resample
from seolet.symbols import (absolute_position, semi_absolute_position, relative_turning_position,
                            relative_direction_position)


class Trj2Txt(BaseEstimator, ClassifierMixin, TransformerMixin):
    RES_TIME = "time"
    RES_TRAVEL = "travel"
    RES_ABSOLUTE = "absolute"

    SYM_ABSOLUTE = "absolute"
    SYM_SEMI_ABSOLUTE = "semi-absolute"
    SYM_RELATIVE_DIR = "relative-dir"
    SYM_RELATIVE_TURN = "relative-turn"

    def __init__(self,
                 resampling_strategy: str = "time", resampling_interval: int | float = 10,
                 symbols_strategy: str = "relative", geohash_precision: int = None, geohash_suffix: int = None,
                 min_radius_meters: int | float = None,
                 n_symbols: int = None, sliding_win_sizes: int = None,
                 model_to_fit: BaseEstimator = None,
                 verbose: bool = False, n_jobs: int = 1):
        self.X_words = None
        self.all_terms = None
        self.term_in_all = None
        self.idf = None
        self.tf = None
        self.tf_idf = None
        if sliding_win_sizes is None:
            sliding_win_sizes = [2]

        if resampling_strategy not in [Trj2Txt.RES_TIME, Trj2Txt.RES_TRAVEL, Trj2Txt.RES_ABSOLUTE]:
            raise ValueError(f"resampling_strategy must be "
                             f"{[Trj2Txt.RES_TIME, Trj2Txt.RES_TRAVEL, Trj2Txt.RES_ABSOLUTE]}")
        if symbols_strategy == Trj2Txt.SYM_ABSOLUTE:
            if geohash_precision is None:
                geohash_precision = 7
            if geohash_suffix is not None or min_radius_meters is not None or n_symbols is not None:
                warnings.warn("n_symbols, geohash_suffix and min_radius will be ignored")

        elif symbols_strategy == Trj2Txt.SYM_SEMI_ABSOLUTE:
            if geohash_precision is None:
                geohash_precision = 7
            if geohash_suffix is None:
                geohash_suffix = 2
            if min_radius_meters is not None or n_symbols is not None:
                warnings.warn("n_symbols and min_radius will be ignored")

        elif symbols_strategy in [Trj2Txt.SYM_RELATIVE_DIR, Trj2Txt.SYM_RELATIVE_TURN]:
            if min_radius_meters is None:
                min_radius_meters = 5
            if n_symbols is None:
                n_symbols = 8 + 1
            if geohash_precision is not None or geohash_suffix is not None:
                warnings.warn("geohash_precision and geohash_suffix will be ignored")
        else:
            raise ValueError(f"symbols_strategy must be "
            f"{[Trj2Txt.SYM_ABSOLUTE, Trj2Txt.SYM_SEMI_ABSOLUTE, Trj2Txt.SYM_RELATIVE_DIR, Trj2Txt.SYM_RELATIVE_TURN]}")

        self.resampling_strategy = resampling_strategy
        self.resampling_interval = resampling_interval
        self.symbols_strategy = symbols_strategy
        self.geohash_precision = geohash_precision
        self.geohash_suffix = geohash_suffix
        self.min_radius = min_radius_meters
        self.n_symbols = n_symbols
        self.sliding_win_sizes = sliding_win_sizes
        self.model_to_fit = model_to_fit
        self.verbose = verbose
        self.n_jobs = n_jobs

    def trj2symb(self, X: Trajectories):
        symbol_X = dict()

        for k, trj in tqdm(X.items(), desc="converting trj to symbols", leave=False, disable=not self.verbose):
            if self.resampling_strategy == self.RES_TIME:
                res_trj = time_resample(trj, time_in_seconds=self.resampling_interval)
            elif self.resampling_strategy == self.RES_TRAVEL:
                res_trj = travel_distance_resample(trj, distance_in_meters=self.resampling_interval)
            else:
                res_trj = absolute_distance_resample(trj, distance_in_meters=self.resampling_interval)

            if self.symbols_strategy == self.SYM_ABSOLUTE:
                sym_trj = absolute_position(res_trj, precision=self.geohash_precision)
            elif self.symbols_strategy == self.SYM_SEMI_ABSOLUTE:
                sym_trj = semi_absolute_position(res_trj, precision=self.geohash_precision,
                                                 suffix_len=self.geohash_suffix)
            elif self.symbols_strategy == self.SYM_RELATIVE_DIR:
                sym_trj = relative_direction_position(res_trj, min_dist=self.min_radius, n_symbols=self.n_symbols)
            else:
                sym_trj = relative_turning_position(res_trj, min_dist=self.min_radius, n_symbols=self.n_symbols)

            symbol_X[k] = sym_trj

        return symbol_X

    def symb2word(self, X: dict):
        X_words = dict()
        for k, v in tqdm(X.items(), desc="Extracting words", leave=False, disable=not self.verbose):
            words = []
            for step_size in self.sliding_win_sizes:
                for i in range(step_size, len(v), step_size):
                    words.append(tuple(v[i - step_size:i]))
            X_words[k] = words

        return X_words

    def get_discriminative_words(self, X: dict):  #X={k:[w1, w2]}
        documents_keys = list(X.keys())

        term_in_document = dict()  # count of t in d
        for k in tqdm(documents_keys, desc="Counting term in documents", leave=False, disable=not self.verbose):
            if k not in term_in_document:
                term_in_document[k] = dict()
            for term in X[k]:
                if term not in term_in_document[k]:
                    term_in_document[k][term] = 0
                term_in_document[k][term] += 1

        n_words_in_document = dict()  #number of words in d
        for k, v in tqdm(term_in_document.items(), desc="Counting words in document", leave=False,
                         disable=not self.verbose):
            n_words_in_document[k] = sum([count for _, count in v.items()])

        if self.term_in_all is None or self.all_terms is None:
            self.all_terms = []
            for k in documents_keys:
                self.all_terms += X[k]
            self.all_terms = set(self.all_terms)

            self.term_in_all = dict()  #df: occurrence of t in documents
            for term in tqdm(self.all_terms, desc="Counting documents containing term", leave=False,
                             disable=not self.verbose):
                self.term_in_all[term] = 0
                for doc in documents_keys:
                    if term in term_in_document[doc]:
                        self.term_in_all[term] += 1

        N = len(X)

        tf_idf = dict()  # doc x term
        tf = dict()  # doc x term
        idf = dict()  # term
        for doc_key in tqdm(documents_keys, desc="computing tf-idf", leave=False, disable=not self.verbose):
            tf_idf[doc_key] = dict()
            tf[doc_key] = dict()
            for term in self.all_terms:
                #tf(t,d) = count of t in d / number of words in d

                _term_in_document = 0
                if term in term_in_document[doc_key]:
                    _term_in_document = term_in_document[doc_key][term]
                _n_words_in_document = max(n_words_in_document[doc_key], 1) #to avoid division by 0
                _term_in_all = self.term_in_all[term]

                _tf = _term_in_document / _n_words_in_document

                tf[doc_key][term] = _tf
                if term not in idf:
                    idf[term] = math.log2(N / (1 + _term_in_all))
                _idf = idf[term]

                tf_idf[doc_key][term] = _tf * _idf

        return tf_idf, tf, idf

    def tfidf_to_np(self, tf_idf):
        terms = []
        for doc_name in tqdm(tf_idf.keys(), desc="Indexing terms", disable=not self.verbose, leave=False):
            for term in tf_idf[doc_name].keys():
                terms.append(term)
        terms = set(terms)

        tfidf_matrix = np.zeros((len(tf_idf), len(terms)))
        for i, doc_name in enumerate(
                tqdm(tf_idf.keys(), desc="Creating the np matrix", disable=not self.verbose, leave=False)):
            for j, term in enumerate(terms):
                if term in tf_idf[doc_name]:
                    tfidf_matrix[i, j] = tf_idf[doc_name][term]

        return tfidf_matrix

    def fit(self, X: Trajectories, y=None):
        X_words, _, _, _ = self.transform(X)

        # by normalizing as such, ed=cosine sim
        length = np.sqrt((X_words ** 2).sum(axis=1))[:, None]
        X_words = X_words / length

        self.X_words = X_words

        if self.model_to_fit is not None:
            self.model_to_fit.fit(X_words, y)

        return self

    def transform(self, X: Trajectories):
        trj_as_symbols = self.trj2symb(X)
        trj_as_words = self.symb2word(trj_as_symbols)

        self.tf_idf, self.tf, self.idf = self.get_discriminative_words(trj_as_words)

        X_words = self.tfidf_to_np(self.tf_idf)

        self.X_words = X_words

        return X_words, self.tf_idf, self.tf, self.idf

    def predict(self, X: Trajectories):
        X_words, _, _, _ = self.transform(X)
        length = np.sqrt((X_words ** 2).sum(axis=1))[:, None]
        X_words = X_words / length
        
        self.X_words = X_words

        if self.model_to_fit is not None:
            return self.model_to_fit.predict(X_words)
        return None


if __name__ == "__main__":
    d0 = 'Geeks for geeks'
    d1 = 'Geeks'
    d2 = 'r2j'

    X = {
        "d0": [tuple([w for w in x]) for x in d0.lower().split()],
        "d1": [tuple([w for w in x]) for x in d1.lower().split()],
        "d2": [tuple([w for w in x]) for x in d2.lower().split()],
    }

    tf_idf, tf, idf = Trj2Txt().get_discriminative_words(X)
    print(tf_idf)
    print(tf)
    print(idf)
