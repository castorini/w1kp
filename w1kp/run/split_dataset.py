import argparse

from matplotlib import pyplot as plt

from w1kp import HitBatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', '-i', required=True, type=str, nargs='+')
    parser.add_argument('--train-pct', type=int, default=90)
    parser.add_argument('--input-folder', '-f', type=str)
    parser.add_argument('--output-suffix', '-o', type=str, default='-split.csv')
    parser.add_argument('--groupby', type=str, default='id', choices=['prompt', 'model', 'id'])
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--axis', type=int, default=0)
    args = parser.parse_args()

    batch = HitBatch.from_csv(
        *args.input_files,
        approved_only=True,
        remove_attention_checks=True,
    )

    if args.train_pct and args.axis == 0:
        train_batch, test_batch = batch.split(train_pct=args.train_pct, groupby=args.groupby, limit=args.limit)
        train_batch.to_csv('train' + args.output_suffix)
        test_batch.to_csv('test' + args.output_suffix)
    else:
        batches = batch.split(groupby=args.groupby, axis=args.axis)

        for idx, batch in enumerate(batches):
            batch.to_csv(f'batch-{idx}' + args.output_suffix)


if __name__ == '__main__':
    main()
