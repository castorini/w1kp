import argparse

import pandas as pd

from w1kp import HitBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str)
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.input_file

    batch = HitBatch.from_csv(args.input_file)

    # Mark workers failing attention checks
    fail_workers_ids = set()
    pass_workers_ids = set(batch.df['WorkerId'].unique().tolist())

    for row in batch:
        data = row.extract_info_from_url()
        fail = False

        if data['ref_seed'] == data['seed1'] and 'Image B' in row['Answer.category.label']:
            fail = True
        elif data['ref_seed'] == data['seed2'] and 'Image A' in row['Answer.category.label']:
            fail = True

        if fail:
            fail_workers_ids.add(row['WorkerId'])

            try:
                pass_workers_ids.remove(row['WorkerId'])
            except KeyError:
                pass

    print(f'Pass percentage: {len(pass_workers_ids) / len(batch.df["WorkerId"].unique()) * 100:.2f}%')

    # Mark rows appropriately
    for wid in fail_workers_ids:
        batch.df.loc[batch.df['WorkerId'] == wid, 'Reject'] = 'One or more attention checks failed.'

    for wid in pass_workers_ids:
        batch.df.loc[batch.df['WorkerId'] == wid, 'Approve'] = 'x'

    batch.df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
