import argparse
from collections import Counter

import pandas as pd

from w1kp import HitBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str)
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.input_file

    df = pd.read_csv(args.input_file)

    # Mark workers failing attention checks
    fail_workers_ids = set()
    pass_workers_ids = set(df['WorkerId'].unique().tolist())
    failures = Counter()

    for _, row in df.iterrows():
        if row['AssignmentStatus'] != 'Submitted':
            continue

        if 'fake' not in row['Input.image_url']:
            continue

        if '(5)' not in row['Answer.category.label']:
            failures[row['WorkerId']] += 1

        if failures[row['WorkerId']] >= 2:
            fail_workers_ids.add(row['WorkerId'])

            try:
                pass_workers_ids.remove(row['WorkerId'])
            except KeyError:
                pass

    print(f'Pass percentage: {len(pass_workers_ids) / len(df["WorkerId"].unique()) * 100:.2f}%')

    # Mark rows appropriately
    for wid in fail_workers_ids:
        df.loc[df['WorkerId'] == wid, 'Reject'] = 'Two or more attention checks failed.'

    for wid in pass_workers_ids:
        df.loc[df['WorkerId'] == wid, 'Approve'] = 'x'

    df.loc[df['AssignmentStatus'] == 'Approved', 'Approve'] = 'x'
    df.loc[df['AssignmentStatus'] == 'Rejected', 'Reject'] = 'One or more attention checks failed.'
    df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
