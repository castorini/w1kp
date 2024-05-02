import argparse
import asyncio
import json
from pathlib import Path
import pandas as pd
import sys

from tqdm.asyncio import tqdm
import aioboto3

from w1kp import Comparison2AFCExperiment


async def amain():
    async def upload_file(bucket, path):
        async with sem:
            await bucket.upload_file(
                str(path),
                str(path.relative_to(args.input_folder)),
                ExtraArgs={'ACL': 'public-read'},
            )

        csv_rows.append(dict(image_url=args.url.format(bucket=args.bucket) + str(path.relative_to(args.input_folder))))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', required=True, type=Path)
    parser.add_argument('--profile', '-p', type=str, default='w1kp')
    parser.add_argument('--bucket', '-b', type=str, default='tetrisdaemon')
    parser.add_argument('--url', '-u', type=str, default='https://{bucket}.s3.amazonaws.com/')
    parser.add_argument('--limit', '-l', type=int, default=1100)
    parser.add_argument('--model', type=str, default='dalle3', choices=['dalle3', 'sdxl', 'sd2', 'imagen', 'midjourney'])
    parser.add_argument('--skip-until-id', type=int, default=0)
    args = parser.parse_args()

    session = aioboto3.Session(profile_name=args.profile)
    sem = asyncio.Semaphore(64)  # limit to 64 open files
    exps = Comparison2AFCExperiment.from_folder(args.input_folder)
    comparison_files = []
    csv_rows = []

    for exp in exps:
        if exp.model_name == args.model:
            comparison_files.append(exp.get_comparison_path())

    comparison_files.sort(key=lambda p: int(p.parent.name))
    num_ids = set()

    for path in comparison_files:
        if int(path.parent.name) < args.skip_until_id:
            continue

        if len(num_ids) >= args.limit:
            break

        num_ids.add(int(path.parent.name))

    comparison_files = [path for path in comparison_files if int(path.parent.name) in num_ids]

    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(args.bucket)

        await tqdm.gather(*(upload_file(bucket, path) for path in comparison_files))

    pd.DataFrame(csv_rows).to_csv(sys.stdout, index=False)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()