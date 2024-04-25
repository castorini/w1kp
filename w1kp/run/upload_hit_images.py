import argparse
import asyncio
from pathlib import Path
import pandas as pd
import sys

from tqdm.asyncio import tqdm
import aioboto3


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
    args = parser.parse_args()

    session = aioboto3.Session(profile_name=args.profile)
    sem = asyncio.Semaphore(64)  # limit to 64 open files
    comparison_files = list(args.input_folder.rglob('**/comparison-*.jpg'))
    csv_rows = []

    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(args.bucket)

        await tqdm.gather(
            *(upload_file(bucket, path) for _, path in zip(range(args.limit), comparison_files)),
            total=min(sum(1 for _ in comparison_files), args.limit),
        )

    pd.DataFrame(csv_rows).to_csv(sys.stdout, index=False)



def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()