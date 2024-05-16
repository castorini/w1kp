import argparse

from w1kp import DreamSimDistanceMeasure, ListwiseDistanceMeasureServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=6555)
    args = parser.parse_args()

    measure = DreamSimDistanceMeasure().to_listwise()
    server = ListwiseDistanceMeasureServer(f'tcp://0.0.0.0:{args.port}', measure)
    server.run()


if __name__ == '__main__':
    main()
