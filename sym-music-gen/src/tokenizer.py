from pathlib import Path
from symusic import Score


import miditok


class Mirex2025Tokenizer(miditok.MusicTokenizer):
    def __init__(self, *args, **kwargs):
        self._max_position = 271
        self._max_pitch = 80
        self._max_duration = 271
        super().__init__(*args, **kwargs)

    def _score_to_tokens(self, score: Score, *args, **kwargs):
        tokens = []
        # Assume single track
        notes = score.tracks[0].notes

        for n in notes:
            # For now if the track is longer than position max_postion, just skip the rest

            if n.time >= self._max_position:
                break

            tokens += [
                f"Position_{n.time}",
                f"Pitch_{n.pitch}",
                f"Duration_{n.duration}",
            ]

        tok_seq = miditok.TokSequence(tokens=tokens)

        self.complete_sequence(tok_seq)

        return [tok_seq]

    def _tokens_to_score(self, tokens, programs):
        raise NotImplementedError()

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        graph = {"Position": {"Pitch"}, "Pitch": {"Duration"}, "Duration": {"Position"}}
        return graph

    def _add_time_events(self, events, time_division: int):
        raise NotImplementedError()

    def _create_base_vocabulary(self):
        vocab = []

        for p in range(self._max_position + 1):
            vocab.append(f"Position_{p}")

        for d in range(self._max_duration + 1):
            vocab.append(f"Duration_{d}")

        for p in range(self._max_pitch + 1):
            vocab.append(f"Pitch_{p}")
        return vocab


file_path = Path.cwd() / "data" / "test2.mid"

config = miditok.TokenizerConfig(
    pitch_range=(0, 127), use_velocities=False, encode_ids_splits="no"
)

remi = miditok.REMI(config)
mirex_custom = Mirex2025Tokenizer()


rt = remi.encode(file_path)
mc = mirex_custom.encode(file_path)

print(f"REMI Tokenization:\t{rt[0].tokens}")
print(f"Number REMI tokens:\t{len(rt[0].tokens)}")
print("\n\n------------------\n\n")
print(f"Mirex 2025 Custom Tokenization:\t{mc[0].tokens}")
print(f"Number Mirex 2025 Custom tokens:\t{len(mc[0].tokens)}")
