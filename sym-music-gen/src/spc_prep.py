import argparse
import json
import os
from pathlib import Path
import random
import tqdm

def prep(percent_holdout: int = 10, minimum_score: float = 0.9):
    # figure out where we are
    dataset_path = Path.cwd()/'data'/'aria-midi-v1-pruned-ext'

    # confirm this exists
    if not dataset_path.is_dir():
        print('Please run get_data.sh!')
        exit(0)

    outpath = Path.cwd()/'data'/'filtered'

    if not outpath.is_dir():
        outpath.mkdir()
        (outpath/'train').mkdir()
        (outpath/'val').mkdir()
    else:
        print('Already split!')
        exit(1)

    # load metadata
    with (dataset_path/'metadata.json').open() as io:
        metadata = json.load(io)

    split = {}

    # todo maybe optimize by reading metadata first, getting every
    # sample_idx, score_idx pair, checking their score. THEN we do a second pass
    # to do the percent split to guarantee a more uniform split

    not_present = 0
    for sample_path in tqdm.tqdm((dataset_path/'data').glob('**/*.mid')):
        sample_idx, score_idx = sample_path.stem.split("_")

        is_train = split.get(sample_idx)

        if is_train is None:
            is_train = random.random() > (percent_holdout/100)
            split[sample_idx] = is_train


        if str(sample_idx) not in metadata:
            #print(f'sample {str(sample_idx)} not in metadata')
            not_present+=1
            continue

        if metadata[str(sample_idx)]['audio_scores'][str(score_idx)] > minimum_score:
            if is_train:
                dest = outpath/'train'/f'{sample_idx}_{score_idx}.mid'
            else:
                dest = outpath/'val'/f'{sample_idx}_{score_idx}.mid'
            os.link(sample_path, dest)
    print(f'filtered data deposited in {outpath}.')
    print(f'number of skipped tracks; not present in metadata: {not_present}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='prep.py'
    )
    parser.add_argument("--minscore",
                        type=float,
                        default=0.9,
                        help="set minimum audio score to accept."
                        )
    parser.add_argument("--holdout",
                        type=int,
                        default=10,
                        help="set percentage of dataset to hold out for \
                        validation."
                        )
    args = parser.parse_args()

    prep(args.holdout, args.minscore)
