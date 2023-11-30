"""
Adapted from the notebook:
https://github.com/google-research/google-research/tree/master/coref_mt5#coreference-resolution-through-a-seq2seq-transition-based-system
"""

import os
import time
from typing import List

import click
from transformers import MT5Tokenizer, T5ForConditionalGeneration

from state import State
from util import (create_document, create_next_batch, extract_result_string,
                  predict_coreferences, read_jsonl, write_jsonl)


@click.command()
@click.option('--input_fname')
@click.option('--tokenizer_path', default='mt5-coref-pytorch/link-append-xxl')
@click.option('--model_path',     default='mt5-coref-pytorch/link-append-xxl')
@click.option('--batch_size',     default=1, type=int)
def main(input_fname, tokenizer_path, model_path, batch_size):

  ### setup model
  tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path, legacy=False)
  model = T5ForConditionalGeneration.from_pretrained(model_path)
  model = model.to(device='cuda')


  ### load ontonotes
  inputs = read_jsonl(input_fname)

  states_dict = {}
  for doc in inputs:
      states_dict[doc['document_id']] = State(create_document(doc), tokenizer)

  xpand_only = False
  total_time = time.time()
  total_results = 0

  debug = True

  num_done = 0
  while num_done < len(states_dict):  # while states
      t = time.time()
      states, batches = create_next_batch(states_dict, batch_size=batch_size, num_batches=1)

      if not states:
          break

      documents_processing = set([x.input_document['doc_key'] for x in states])

      if debug:
        print(f'Processing documents: {documents_processing}')

      predictions = predict_coreferences(tokenizer, model, batches, len(batches))
      results = extract_result_string(predictions)
      
      if debug:
        print(predictions)
        print(results)

      for state, result, batch in zip(states, results, batches):
          state.extend(result)

          if debug:
              print('input batch[0]: ', batch)
              print('mt5 output:     ', results)
      
      total_results += len(results)
      if debug:
          print(
              f'time { time.time()-t}, round time/seq : {(time.time()-t)/len(results)}'
              f' total time/seq: {(time.time()-total_time)/total_results}'
          )



if __name__ == '__main__':
    main()
