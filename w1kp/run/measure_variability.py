import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from w1kp import GenerationExperiment, ListwiseDistanceMeasureClient, query_inverted_cdf


def main():
    def do_measure(measure, prompt, experiments, seed_map):
        try:
            images = [exp.load_image() for exp in experiments]
            images = list(filter(lambda x: x is not None, images))

            if len(images) <= 1:
                return {}

            ret = measure.measure(prompt, images, reduce='reusability' if args.run_reusability_analysis else 'all')

            if args.cdf_xy_path:
                ret = {k: d if 'idx' in k else query_inverted_cdf(cdf_x, cdf_y, d) for k, d in ret.items()}

            for k, d in list(ret.items()):
                if 'idx' in k:
                    ret[k] = tuple([seed_map[idx] for idx in d])

            print(prompt, ret)
            ret['prompt'] = prompt

            return ret
        except:
            return {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--output-suffix', '-o', type=str, default='output.csv')
    parser.add_argument('--cdf-xy-path', '-c', type=str)
    parser.add_argument('--async-listwise-urls', '-al', type=str, nargs='+')
    parser.add_argument('--run-reusability-analysis', '-rra', action='store_true')
    parser.add_argument('--min-images', '-mi', type=int, default=2)
    parser.add_argument('--max-num', type=int, default=1000000)
    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    if args.cdf_xy_path:
        cdf_x, cdf_y = torch.load(args.cdf_xy_path)

    measures = [ListwiseDistanceMeasureClient(url) for url in args.async_listwise_urls]

    routines = []
    g = GenerationExperiment.iter_by_id(input_folder, model_name=args.model)

    for idx, experiments in tqdm(enumerate(g)):
        if idx >= args.max_num:
            g.send(True)  # send True to close the generator
            break

        if len(experiments) < args.min_images:
            continue

        measure = measures[idx % len(measures)]
        seed_map = {idx2: exp.seed for idx2, exp in enumerate(experiments)}
        routines.append(joblib.delayed(do_measure)(measure, experiments[0].prompt, experiments, seed_map))

    results = joblib.Parallel(n_jobs=8 if args.run_reusability_analysis else 16)(tqdm(routines))
    results = list(filter(lambda x: x, results))

    df = pd.DataFrame(results)
    df.to_csv(input_folder.name + '-' + args.model + '-' + args.output_suffix, index=False)
    df = df.drop(columns='prompt')
    print(df.mean())


if __name__ == '__main__':
    main()
