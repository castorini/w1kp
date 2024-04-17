from functools import lru_cache
from typing import Literal, List

import nltk

__all__ = ['GroupedLexicon']


class GroupedLexicon:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.group2words = kwargs
        self.word2group = {word: group for group, words in kwargs.items() for word in words}
        self.classify = lru_cache(maxsize=None)(self.classify)
        self._group2words_sets = None

    def find_group(self, word: str) -> str:
        return self.word2group[word]

    def list_words(self, group: str) -> List[str]:
        return self.group2words[group]

    def classify(self, word: str) -> str | None:
        if self._group2words_sets is None:
            self._group2words_sets = {group: set(words) for group, words in self.group2words.items()}

        for group, words in self._group2words_sets.items():
            if word in words:
                return group

        return None

    def __str__(self):
        return f'{self.name} ({len(self.group2words)} groups)'

    def __repr__(self):
        return str(self)

    @classmethod
    def from_pretrained(cls, name: Literal['opinion_lexicon'] = 'opinion_lexicon') -> 'GroupedLexicon':
        match name:
            case 'opinion_lexicon':
                nltk.download('opinion_lexicon')
                pos_group = nltk.corpus.opinion_lexicon.positive()
                neg_group = nltk.corpus.opinion_lexicon.negative()

                return cls(name, positive=pos_group, negative=neg_group)
