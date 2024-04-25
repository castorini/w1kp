import argparse
import asyncio
import itertools
import math

import numpy as np
from tqdm import tqdm

from w1kp import GenerationExperiment, LPIPSDistanceMeasure


async def amain():
    models = ['sdxl', 'sd2', 'dalle3', 'imagen', 'midjourney']

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, default='output')
    parser.add_argument('--model', type=str, default='dalle3', choices=models)
    parser.add_argument('--id', type=str, default='0')
    parser.add_argument('--action', '-a', type=str, default='debug', choices=['debug', 'run'])
    args = parser.parse_args()

    if args.action == 'debug':
        experiments1 = list(GenerationExperiment.iter_by_seed(args.input_folder, '0', model_name=args.model))
        experiments2 = list(GenerationExperiment.iter_by_seed(args.input_folder, '1', model_name=args.model))

        measure = LPIPSDistanceMeasure()
        within_measures1 = []
        within_measures2 = []
        across_measures = []
        debug_strings = []

        for idx1, idx2 in itertools.product(range(len(experiments1)), range(len(experiments2))):
            exp1 = experiments1[idx1]
            exp2 = experiments2[idx2]
            across_measures.append(measure(exp1.prompt, exp1.load_image(), exp2.load_image()))

        for idx1, idx2 in itertools.combinations(range(len(experiments1)), 2):
            exp1 = experiments1[idx1]
            exp2 = experiments1[idx2]
            within_measures1.append(measure(exp1.prompt, exp1.load_image(), exp2.load_image()))
            debug_strings.append((f'({idx1}, {idx2}):', within_measures1[-1]))

        debug_strings.sort(key=lambda x: x[1])

        for debug_string in debug_strings:
            print(*debug_string)

        for idx1, idx2 in itertools.combinations(range(len(experiments2)), 2):
            exp1 = experiments2[idx1]
            exp2 = experiments2[idx2]
            within_measures2.append(measure(exp1.prompt, exp1.load_image(), exp2.load_image()))

        print(f'Within 1: {np.mean(within_measures1)}')
        print(f'Within 2: {np.mean(within_measures2)}')
        print(f'Across: {np.mean(across_measures)}')


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
