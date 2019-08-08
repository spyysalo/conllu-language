#!/usr/bin/env python3

import sys
import os

from logging import warning
from collections import OrderedDict, Counter


# Number of languages for past sentences to remember
MAX_HISTORY = 3


# Universal Dependencies languages with MUSE vectors
UD_MUSE_LANGUAGES = [
    "ar", "bg", "ca", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "he", "hu", "id", "it", "no", "pl", "pt", "ro", "ru", "sk", "sl",
    "es", "sv", "tr", "uk", "vi",
]


# CoNLL-U comment identifying sentence text
TEXT_COMMENT = '# text = '


# CoNLL-U comment identifying new document
NEWDOC_COMMENT = '# newdoc'


class Detector(object):
    def __init__(self):
        self.history = []
        self.threshold = 0

    def detect(self, text):
        ranked = self.rank_languages(text)
        language, prob = ranked[0]
        if prob <= self.threshold and self.history:
            # highest ranked has low probability, fall back on most frequent
            language = Counter(self.history).most_common(1)[0][0]
        else:
            self.history.append(language)
            self.history = self.history[-MAX_HISTORY:]
        return language

    def rank_languages(self, text):
        raise NotImplementedError

    def set_languages(self, languages):
        raise NotImplementedError

    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def reset(self):
        self.history = []


class LangdetectDetector(Detector):
    module = None
    factory = None
    
    def __init__(self):
        super(LangdetectDetector, self).__init__()
        if LangdetectDetector.module is None:
            LangdetectDetector.module = __import__('langdetect')
            factory = LangdetectDetector.module.DetectorFactory()
            factory.load_profile(LangdetectDetector.module.PROFILES_DIRECTORY)
            LangdetectDetector.factory = factory
        self.languages = None

    def rank_languages(self, text, cutoff=None):
        detector = LangdetectDetector.factory.create()
        detector.seed = 1234    # random seed (consistency)
        threshold = cutoff if cutoff is not None else 1e-10
        detector.PROB_THRESHOLD = threshold    # for get_probabilities()
        detector.append(text)
        if self.languages is not None:
            prior_map = { l: 1.0/len(self.languages) for l in self.languages }
            detector.set_prior_map(prior_map)
        try:
            probs = detector.get_probabilities()
            probs = [(p.lang, p.prob) for p in probs]
            if cutoff is not None:
                probs = [(l,p) for l,p in probs if p > cutoff]
        except:
            warning('failed to detect language for input "{}"'.format(text))
            probs = [ ('en', 0.0) ]    # fallback
        return probs

    def set_languages(self, languages):
        self.languages = languages[:]


class LangidDetector(Detector):
    module = None
    identifier = None

    def __init__(self):
        super(LangidDetector, self).__init__()
        if LangidDetector.module is None:
            langid = __import__('langid.langid').langid
            LangidDetector.module = langid
            identifier = langid.LanguageIdentifier.from_modelstring(
                langid.model, norm_probs=True)
            LangidDetector.identifier = identifier

    def rank_languages(self, text, cutoff=None):
        probs = LangidDetector.identifier.rank(text)
        if cutoff is not None:
            probs = [(l,p) for l,p in probs if p > cutoff]
        return probs

    def set_languages(self, languages):
        LangidDetector.identifier.set_languages(languages)


DETECTOR_MAP = OrderedDict([
    ('langdetect', LangdetectDetector),
    ('langid', LangidDetector),
])


DETECTORS = list(DETECTOR_MAP.keys())


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-l', '--limit-langs', default=False, action='store_true',
                    help='limit language options to 29 UD/MUSE languages')
    ap.add_argument('-t', '--threshold', default=0, type=float,
                    help='confidence threshold for changing language')
    ap.add_argument('-p', '--prefix', default=False, action='store_true',
                    help='prexif FORM and LEMMA with language code')
    ap.add_argument('-u', '--use', default=DETECTORS[0], choices=DETECTORS,
                    help='language ID module (default {})'.format(DETECTORS[0]))
    ap.add_argument('conllu', nargs='+', help='CoNLL-U data')
    return ap


class FormatError(Exception):
    pass


class Word(object):
    def __init__(self, id_, form, lemma, upos, xpos, feats, head, deprel,
                 deps, misc):
        self.id = id_
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def __str__(self):
        return '\t'.join([
            self.id, self.form, self.lemma, self.upos, self.xpos, self.feats,
            self.head, self.deprel, self.deps, self.misc
        ])


class Sentence(object):
    def __init__(self, comments, words, source, lineno):
        self.comments = comments
        self.words = words
        self.source = source
        self.lineno = lineno
        self._text = None

    @property
    def text(self):
        if self._text is None:
            text_lines = [
                s for s in self.comments if s.startswith(TEXT_COMMENT)
            ]
            if not text_lines:
                raise ValueError('no "# text" line: {} line {}'\
                                 .format(self.source, self.lineno))
            elif len(text_lines) > 1:
                raise ValueError('multiple "# text" lines: {} line {}'\
                                 .format(self.source, self.lineno))
            self._text = text_lines[0][len(TEXT_COMMENT):]
        return self._text

    def __str__(self):
        return '\n'.join(self.comments + [str(w) for w in self.words] + [''])


class Conllu(object):
    def __init__(self, path):
        self.path = path
        self.stream = open(path)
        self.lineno = 0
        self.finished = False
        self.current = self._get_sentence()

    def __iter__(self):
        return self

    def __next__(self):
        s = self.current
        if s is None:
            raise StopIteration()            
        self.current = self._get_sentence()
        return s

    def next(self):
        return self.__next__()

    def _get_sentence(self):
        if self.finished:
            return None
        start_lineno = self.lineno + 1
        comments, words = [], []
        for l in self.stream:
            self.lineno += 1
            l = l.rstrip('\n')
            if not l or l.isspace():
                # blank line marks end of sentence
                if not words:
                    raise FormatError('empty sentence on line {} in {}'.format(
                        self.lineno, self.path))
                s = Sentence(comments, words, self.path, start_lineno)
                return s
            elif l.startswith('#'):
                comments.append(l)
            else:
                fields = l.split('\t')
                if len(fields) != 10:
                    raise FormatError(
                        'expected 10 tab-separated fields, got {} on line {}'\
                        'in {}, got {}: {}'.format(
                            self.lineno, self.path, len(fields), l))
                words.append(Word(*fields))
        self.finished = True
        return None

            
def get_detector(args):
    if args.use not in DETECTOR_MAP:
        raise ValueError(args.use)
    detector = DETECTOR_MAP[args.use]()
    if args.limit_langs:
        detector.set_languages(UD_MUSE_LANGUAGES)
    detector.set_threshold(args.threshold)
    return detector


def main(argv):
    args = argparser().parse_args(argv[1:])
    detector = get_detector(args)
    for fn in args.conllu:
        conllu = Conllu(fn)
        for s in conllu:
            if any(c.startswith(NEWDOC_COMMENT) for c in s.comments):
                detector.reset()    # forget history
            language = detector.detect(s.text)
            ranked = detector.rank_languages(s.text, cutoff=1e-5)
            s.comments.append('# language = {}'.format(language))
            s.comments.append('# lang_probs = {}'.format(
                ' '.join(['{}:{}'.format(l,p) for l,p in ranked])))
            if args.prefix:
                for w in s.words:
                    w.form = '^{}:{}'.format(language, w.form)
                    w.lemma = '^{}:{}'.format(language, w.lemma)
            print(s)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
