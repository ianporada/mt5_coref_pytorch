# mt5_coref_pytorch

**Update: This is a minimal example using code based on the released notebook. You can find an example of how to run inference on an entire dataset using our reimplementation here: [ianporada/coref-reeval/models/LinkAppend](https://github.com/ianporada/coref-reeval/tree/main/models/LinkAppend).**

## About 

This repo contains code for running [Bohnet et al.'s MT5 seq2seq coreference resolution model](https://github.com/google-research/google-research/tree/master/coref_mt5#coreference-resolution-through-a-seq2seq-transition-based-system) using HuggingFace Transformers. The main processing code comes from the original model's jupyter notebook.

I've converted the released t5x checkpoint to HuggingFace using [convert_t5x_checkpoint_to_flax.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/convert_t5x_checkpoint_to_flax.py). The HuggingFace model is available as ['mt5-coref-pytorch/link-append-xxl'](https://huggingface.co/mt5-coref-pytorch/link-append-xxl).

## Running inference

```bash
python main.py --input_fname input.jsonl
```

where each line in `input.jsonl` is of the form
```json
{
    'document_id': 'example_doc',
    'sentences': [
        {
            'speaker': 'example_speaker',
            'words': ['The', 'thing', 'is', ...]
        },
        ...
    ]
}
```

## Environment

Tested with:

* python=3.10
* pytorch=2.1
* transformers=4.32
* click
