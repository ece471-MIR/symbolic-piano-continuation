import argparse
import tqdm
import torch
import symusic
import miditok
import transformers
import json

from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("--num-generations", type=int, default=4)

    return parser.parse_args()


def notes_to_score(notes):
    score = symusic.Score(ttype=symusic.TimeUnit.quarter)

    track = symusic.Track(ttype=symusic.TimeUnit.quarter)

    for n in notes:
        # In symusic the note duration is specified in quarters. 
        # In the MIREX prompt it is specified in sixteenth notes
        note = symusic.Note(
            time=n["start"] / 4,
            duration=n["duration"] / 4,
            pitch=n["pitch"],
            velocity=90,
            ttype=symusic.TimeUnit.quarter,
        )
        track.notes.append(note)

    score.tracks.append(track)

    score = score.to("tick")

    return score


def main():
    args = __parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model).eval()

    tokenizer = miditok.REMI.from_pretrained(args.model)

    with args.prompt.open() as io:
        prompt = json.load(io)

    prompt_score = notes_to_score(prompt["prompt"])

    prompt_tokens = tokenizer.encode(prompt_score)[0].ids

    prompt_tokens_pt = torch.tensor(prompt_tokens).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_tokens_pt)

    output_dir = Path(f"{args.prompt.stem}_generations")
    output_dir.mkdir(exist_ok=True)

    for n in tqdm.tqdm(range(1, args.num_generations + 1)):
        with torch.no_grad():
            output = model.generate(
                prompt_tokens_pt,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.4,
                do_sample=True,
            )
        full_score = tokenizer.decode(output)

        full_score.dump_midi(str(output_dir / f"sample_{n:02d}.mid"))


if __name__ == "__main__":
    main()
