__all__ = ['HitBatch']

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Any, List, Tuple

import pandas as pd

from .dataset import LPIPSDataset
from .experiment import Comparison2AFCExperiment


def extract_info_from_url(url: str) -> Dict[str, str]:
    return HitRow.patt.match(url).groupdict()


class HitRow:
    patt = re.compile(r'^.*com/(?P<model>.+?)/(?P<id>.+?)/comparison-(?P<seed1>.+?)-(?P<seed2>.+?)-(?P<ref_seed>.+?).jpg$')

    def __init__(self, row: pd.Series):
        self.row = row

    def __getitem__(self, item):
        return self.row[item]

    def __getattr__(self, item):
        return self.row[item]

    def __str__(self):
        return self.row.__str__()

    def __repr__(self):
        return self.row.__repr__()

    def extract_info_from_url(self) -> Dict[str, str]:
        return self.patt.match(self.row['Input.image_url']).groupdict()

    @staticmethod
    def extract_info_from_url_(url: str) -> Dict[str, str]:
        return HitRow.patt.match(url).groupdict()

    @property
    def choice(self):
        return 'a' if 'Image A' in self.row['Answer.category.label'] else 'b'

    def load_comparison_experiment(self, root_folder: str | Path) -> 'Comparison2AFCExperiment':
        return Comparison2AFCExperiment(
            model_name=self.row['model'],
            id=self.row['id'],
            ref_seed=self.row['ref_seed'],
            seed1=self.row['seed1'],
            seed2=self.row['seed2'],
            root_folder=Path(root_folder)
        )


class HitBatch:
    def __init__(self, dataframe: pd.DataFrame):
        self.old_df = self.df = dataframe
        infos = []

        for row in self:
            infos.append(row.extract_info_from_url())

        info_df = pd.DataFrame(infos)
        self.df = pd.concat([self.df, info_df], axis=1)
        self.info_columns = info_df.columns

    def split(self, train_pct: int) -> Tuple['HitBatch', 'HitBatch']:
        old_state = random.getstate()
        random.seed(0)
        train_rows, test_rows = [], []

        df = self.df.groupby(['model', 'id'])

        for _, group in df:
            data = train_rows if 100 * random.random() < train_pct else test_rows

            for _, row in group.iterrows():
                dict_ = row.to_dict()
                del dict_['index']

                for col in self.info_columns:
                    del dict_[col]

                data.append(dict_)

        random.setstate(old_state)

        return HitBatch(pd.DataFrame(train_rows)), HitBatch(pd.DataFrame(test_rows))

    @classmethod
    def from_csv(cls, path: str, approved_only: bool = False, remove_attention_checks: bool = False):
        def is_attention_check(url: str) -> bool:
            d = extract_info_from_url(url)
            return d['seed1'] == d['ref_seed'] or d['seed2'] == d['ref_seed']

        df = pd.read_csv(path)

        if approved_only:
            df = df[df['AssignmentStatus'] == 'Approved']

        if remove_attention_checks:
            df = df[~df['Input.image_url'].apply(is_attention_check)]

        df = df.reset_index()

        return cls(df)

    def to_lpips_dataset(self, image_root_folder: Path) -> 'LPIPSDataset':
        exps = []

        for rows in self.iter_group_by_seed():
            exp = rows[0].load_comparison_experiment(image_root_folder)
            choices = [row.choice for row in rows]
            a_pct = choices.count('a') / len(choices)
            exp.judgement = a_pct
            exps.append(exp)

        return LPIPSDataset.from_experiments(exps)

    def iter_by_row(self) -> Generator[HitRow, Any, None]:
        return iter(self)

    def iter_group_by_seed(self) -> Generator[List[HitRow], Any, None]:
        df = self.df.groupby(['model', 'id', 'ref_seed', 'seed1', 'seed2'])

        for _, group in df:
            yield [HitRow(row) for _, row in group.iterrows()]

    def __iter__(self) -> Generator[HitRow, Any, None]:
        return (HitRow(x) for _, x in self.df.iterrows())
