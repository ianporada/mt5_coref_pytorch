"""
Adapted from the notebook:
https://github.com/google-research/google-research/tree/master/coref_mt5#coreference-resolution-through-a-seq2seq-transition-based-system
"""

import json

import torch


def normalize_speaker(speaker_in):
  """Add '_' before and after speaker name if it does not contain it already"""
  if speaker_in == '-' or speaker_in == '__':
    return '_'

  speaker = speaker_in.replace(' ', '_')
  speaker = speaker.strip()

  if not speaker.startswith('_'):
    speaker = '_'+speaker
  if not speaker.endswith('_'):
    speaker = speaker+'_'
  return speaker


def extract_result_string(predictions):
  """Extract the results from prediction."""
  results = []
  for resp in  predictions:
    # print(resp)
    # if isinstance(resp, list):
    # response should be a list of predictions
    output_text = [resp]
    scores = [1.0 for _ in resp]

    for text, score in zip(output_text, scores):
      # print(text)
      # text = text[0].decode('utf-8')
      text = text[0]
      results.append(text)
  return results


def read_jsonl(fname):
    with open(fname) as f:
        return [json.loads(line) for line in f]


def write_jsonl(fname, examples):
    with open(fname, 'w') as f:
            for ex in examples:
                json.dump(ex, f)
                f.write('\n')



def batch_model_input_to_output(tokenizer, model, inputs, max_new_tokens=384):
    batched_inputs = tokenizer(inputs, padding='longest', return_tensors='pt')
    for k, v in batched_inputs.items():
        batched_inputs[k] = v.to('cuda')
        
    with torch.no_grad():
        generated_ids = model.generate(**batched_inputs, max_new_tokens=max_new_tokens)
        
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return generated_texts


def create_document(document_dict):
  input_document = {
      'doc_key': document_dict['document_id'],
      'sentences': {},
      'token_maps': {},
      'tid_to_word_idx': {},
      'speakers': [],
      'genres': []
  }

  tid = 0
  for k, sentence in enumerate(document_dict['sentences']):
    input_document['sentences'][k] = sentence['words']
    input_document['token_maps'][k] = []

    for word_idx, word in enumerate(input_document['sentences'][k]): # for each word
      input_document['token_maps'][k].append(tid)
      input_document['tid_to_word_idx'][tid] = (k, word_idx)
      
      speaker = sentence['speaker'] if sentence['speaker'] else '_'
      input_document['speakers'].append((tid, speaker))
      input_document['genres'].append(document_dict['document_id'][:2])
      tid += 1
  return input_document


def create_next_batch(states_dict, batch_size=1, num_batches=1):
  batches = [[]]
  states = []
  for key, state in states_dict.items():
    if state.extend_done():
      continue

    states.append(state)
    if len(states) >= (batch_size * num_batches):
      break
  for state in states:
    batches[-1].append(state.get_input_annotation())
    if len(batches[-1]) >= batch_size:
      if len(batches) >= num_batches:
        break
      batches.append([])
  return states, batches


def predict_coreferences(tokenizer, model, batches, num_batches=None):
  if num_batches is not None:
    assert len(batches) == num_batches
  validation_examples = []
  predictions_list = []
  for batch in batches:
    for b in batch:
      validation_examples.append({'input': b})
    
    inputs = batch
    predictions = batch_model_input_to_output(tokenizer, model, inputs)
    
    predictions_list.append(predictions)
  return predictions_list

