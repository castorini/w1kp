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
    args = parser.parse_args()

    session = aioboto3.Session(profile_name=args.profile)
    sem = asyncio.Semaphore(64)  # limit to 64 open files

    df = pd.read_csv(args.input_folder / 'magnitude.csv')
    comparison_files = []
    csv_rows = []

    for _, row in df.iterrows():
        comparison_files.append(args.input_folder / row['image_url'])

    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(args.bucket)
        await tqdm.gather(*(upload_file(bucket, path) for path in comparison_files))

    pd.DataFrame(csv_rows).to_csv(sys.stdout, index=False)


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()