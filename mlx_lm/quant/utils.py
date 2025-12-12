# Copyright © 2025 Apple Inc.

from pathlib import Path
from datasets import load_dataset
import mlx.core as mx


def load_data(tokenizer, num_samples: int, sequence_length: int) -> mx.array:
    save_dir = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
    if not save_dir.exists():
        from urllib import request

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        url = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
        request.urlretrieve(url, save_dir)
    with open(save_dir) as fid:
        texts = fid.read()
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # select random non-overlapping chunks
    tokens = tokens[: (tokens.size // sequence_length) * sequence_length]
    tokens = tokens.reshape(-1, sequence_length)
    segments = mx.random.permutation(tokens.shape[0])
    if num_samples > 0:
        segments = segments[:num_samples]
    return tokens[segments]


def load_audio_data(
    data: str,
    num_samples: int,
    sequence_length: int,
    split="validation",
):
    dataset = load_dataset(data, split=split)
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.select_columns(
        ["text_ids", "speak_ids", "listen_ids"]
    )

    input_list = []
    speak_list = []
    listen_list = []
    max_input_len = 4096
    for data in dataset:
        input_list.append(mx.array(data["text_ids"])[:, :max_input_len])
        speak_list.append(mx.array(data["speak_ids"])[:, :max_input_len])
        listen_list.append(mx.array(data["listen_ids"])[:, :max_input_len])

    input_ids = mx.concat(input_list, axis=1)
    speak_ids = mx.concat(speak_list, axis=1)
    listen_ids = mx.concat(listen_list, axis=1)

    n_split = input_ids.shape[1] // sequence_length
    print(f" * Split into {n_split} blocks")

    input_list = [
        input_ids[:, i * sequence_length : (i + 1) * sequence_length] for i in range(n_split) if i < num_samples
    ]
    speak_list = [
        speak_ids[:, i * sequence_length : (i + 1) * sequence_length] for i in range(n_split) if i < num_samples
    ]
    listen_list = [
        listen_ids[:, i * sequence_length : (i + 1) * sequence_length] for i in range(n_split) if i < num_samples
    ]

    return {
        "text_ids": mx.concat(input_list, axis=0),
        "speak_ids": mx.concat(speak_list, axis=0),
        "listen_ids": mx.concat(listen_list, axis=0),
    }
