import argparse
import multiprocessing as mp
from pathlib import Path

from w1kp import GenerationExperiment, synthesize_2afc_experiments


class MPAFCSerializer:
    lock = mp.Lock()
    read_q = mp.Queue()
    write_q = mp.Queue()

    def __init__(self, input_folder: Path, model_name: str, num_samples: int):
        self.input_folder = input_folder
        self.model_name = model_name
        self.num_samples = num_samples
        self.processes = [mp.Process(target=self._worker, daemon=True) for _ in range(mp.cpu_count())]

        for p in self.processes:
            p.start()

    def generate_all(self):
        ids = GenerationExperiment.get_ids(self.input_folder, model_name=self.model_name)

        for gen_id in ids:
            self.read_q.put(gen_id)

        num_fin = 0
        sum_ = len(ids)

        while num_fin != sum_:
            self.write_q.get()
            num_fin += 1
            print(f'Progress: {num_fin}/{sum_}        ', end='\r')

    def _worker(self):
        while True:
            gen_id = self.read_q.get()
            exps = list(GenerationExperiment.iter_by_seed(self.input_folder, gen_id, self.model_name))
            afc_exps = list(synthesize_2afc_experiments(exps, num_samples=self.num_samples, strategy='random', attention_check=False))
            afc_exps.extend(synthesize_2afc_experiments(exps, num_samples=1, strategy='random', attention_check=True))

            for afc_exp in afc_exps:
                afc_exp.create_comparison_image()

                with self.lock:
                    afc_exp.save()

            self.write_q.put(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', required=True, type=Path)
    parser.add_argument('--clear', '-c', action='store_true')
    parser.add_argument('--num-samples', '-n', type=int, default=10)
    parser.add_argument('--model', '-m', type=str, default='dalle3')
    parser.add_argument('--clear-only', '-C', action='store_true')
    args = parser.parse_args()

    if args.clear or args.clear_only:
        (args.input_folder / '2afc.jsonl').unlink(missing_ok=True)

        for pth in args.input_folder.glob('**/comparison-*.jpg'):
            pth.unlink()

    if args.clear_only:
        return

    serializer = MPAFCSerializer(args.input_folder, args.model, args.num_samples)
    serializer.generate_all()


if __name__ == '__main__':
    main()
