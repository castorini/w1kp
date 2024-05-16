__all__ = ['RGBColorFeatureExtractor', 'LexicalFeatureExtractor', 'GPT4FeatureExtractor']

import asyncio
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from openai import AzureOpenAI
import requests
import stanza
from PIL.Image import Image
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset


@lru_cache(maxsize=100000)
def conceptnet_get_isa(word):
    return len(requests.get(
        f'http://conceptnet5.media.mit.edu/data/5.2/search?rel=/r/IsA&end=/c/en/{word}&limit=1000',
        verify=False
    ).json()['edges'])


class GPT4FeatureExtractor:
    def __init__(self, azure_client: AzureOpenAI):
        self.azure_client = azure_client

    def extract_thematic_roles(self, sentence: str) -> int:
        content = self.generate_gpt4_response(f'Extract the number of semantic thematic roles from this sentence: "{sentence}" Do not explain; only reply with an integer.')

        try:
            return int(content)
        except:
            return 1

    def generate_gpt4_response(self, prompt: str) -> str:
        r = self.azure_client.chat.completions.create(
            messages=[dict(role='user', content=prompt)],
            timeout=60,
            model='gpt-4-32k',
            temperature=0,
        )

        return r.choices[0].message.content


class LexicalFeatureExtractor:
    def __init__(
            self,
            google_corpus_path: str,
            concreteness_corpus_path: str,
            wn_binary_path: str = None
    ):
        """
        Args:
            wn_binary_path: If specified, use this WordNet binary instead of the NLTK WordNet. Needed for custom senses.
        """
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')
        freq_words = list(dict.fromkeys([x.lower().strip() for x in Path(google_corpus_path).read_text().splitlines()]))
        freqs = 1 / np.arange(2, len(freq_words) + 2)
        self.freq_df = pd.DataFrame({'word': freq_words, 'freq': freqs})
        self.freq_df.set_index('word', inplace=True)

        self.concrete_df = pd.read_csv(concreteness_corpus_path, sep='\t')
        self.concrete_df.set_index('Word', inplace=True)
        self.wn_binary_path = wn_binary_path

    @lru_cache(maxsize=100000)
    def cached_nlp(self, text: str):
        return self.nlp(text)

    def count_synsets(self, word: str) -> int:
        def parse_num_senses(output: str) -> int:
            for line in output.splitlines():
                if m := re.match(r'^(\d+) senses of .*$', line):
                    return int(m.group(1))

            return 0

        if self.wn_binary_path:
            num_senses = 0

            for arg in ('-synsn', '-synsa', '-synsv'):
                # Make a subprocess call to the WordNet binary to find noun, adjective, and verb senses
                p = subprocess.Popen([self.wn_binary_path, word, arg], stdout=subprocess.PIPE)
                num_senses += parse_num_senses(p.stdout.read().decode('utf-8'))

            return num_senses
        else:
            return len([s for s in wordnet.synsets(word) if sum(l.count() for l in s.lemmas()) > 1])

    def concreteness(self, word: str) -> float:
        return self.concrete_df.loc[word, 'Conc.M']

    def frequency(self, word: str) -> float:
        return self.freq_df.loc[word.lower().strip(), 'freq']

    def count_hyponyms(self, word: str = None, synsets: List[Synset] = None, is_first_call: bool = True) -> int:
        # wordnet.synsets(word)[0].hyponyms() is not as precise or comprehensive
        return conceptnet_get_isa(word)

    def extract_word(self, word: str) -> Dict[str, float]:
        return dict(
            num_synsets=self.count_synsets(word),
            concreteness=self.concreteness(word),
            frequency=self.frequency(word),
            num_hyponyms=self.count_hyponyms(word),
        )

    def extract_sentence(self, sentence: str) -> Dict[str, float]:
        doc = self.cached_nlp(sentence)
        words = [word.text for sent in doc.sentences for word in sent.words]
        data_dict = defaultdict(list)

        for word in words:
            try:
                data_dict['concreteness'].append(self.concreteness(word))
            except:
                pass

            try:
                data_dict['frequency'].append(self.frequency(word))
            except:
                pass

            try:
                data_dict['num_synsets'].append(self.count_synsets(word))
            except:
                pass

        return {k: np.mean(v) for k, v in data_dict.items()}


class RGBColorFeatureExtractor:
    def __init__(self, image: Image):
        self.image = image

    def mean_area(self, channel: str, hue_tolerances: List[int] = [15, 30, 40], saturation_threshold: int = 10) -> float:
        match channel:
            case 'r' | 'red':
                channel_hue_center = 0
                channel = 0
            case 'g' | 'green':
                channel_hue_center = 120
                channel = 1
            case 'b' | 'blue':
                channel_hue_center = 220
                channel = 2
            case _:
                raise ValueError(f'Invalid channel {channel}')

        pixels = np.array(self.image.getdata())

        # RGB to HSL
        pixels = pixels / 255
        cmax = np.max(pixels, axis=1)
        cmin = np.min(pixels, axis=1)
        delta = cmax - cmin

        # Hue
        hue = np.zeros(len(pixels))
        hue[cmax == cmin] = 0
        hue[cmax == pixels[:, 0]] = 60 * (((pixels[cmax == pixels[:, 0], 1] - pixels[cmax == pixels[:, 0], 2]) / delta[cmax == pixels[:, 0]]) % 6)
        hue[cmax == pixels[:, 1]] = 60 * (((pixels[cmax == pixels[:, 1], 2] - pixels[cmax == pixels[:, 1], 0]) / delta[cmax == pixels[:, 1]]) + 2)
        hue[cmax == pixels[:, 2]] = 60 * (((pixels[cmax == pixels[:, 2], 0] - pixels[cmax == pixels[:, 2], 1]) / delta[cmax == pixels[:, 2]]) + 4)

        # Saturation
        saturation = np.zeros(len(pixels))
        saturation[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

        mask = np.zeros(len(pixels))

        if channel == 0:
            mask[np.abs(((hue + 180) % 360) - (channel_hue_center + 180)) <= hue_tolerances[channel]] = 1
        else:
            mask[np.abs(hue - channel_hue_center) <= hue_tolerances[channel]] = 1

        mask[saturation < saturation_threshold / 100] = 0

        return np.mean(mask)
