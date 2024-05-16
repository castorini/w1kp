import argparse

from w1kp.model.image_generator import AsyncSDXLServer, StableDiffusionXLImageGenerator


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5555)
    parser.add_argument('--device', '-d', type=int, default=0)
    args = parser.parse_args()

    generator = StableDiffusionXLImageGenerator(args.device)
    await AsyncSDXLServer(f'tcp://0.0.0.0:{args.port}', generator).run()


def main():
    import asyncio
    asyncio.run(amain())


if __name__ == '__main__':
    main()