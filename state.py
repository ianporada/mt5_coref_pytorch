"""
Document state adapted from the original notebook:
https://github.com/google-research/google-research/tree/master/coref_mt5#coreference-resolution-through-a-seq2seq-transition-based-system
"""

from matching import get_mentions_for_link_state, match_link_state
from util import normalize_speaker


class State(object):
  """Document state."""

  def __init__(self, input_document, tokenizer, node_wise=True, max_len_doc=3000):
    """ Create State object to process documents.

    Args:
      input_document: dictonary with the input document.
      node_wise: Predict mentions too.
      max_len_doc: max sentence pieace tokens, eg. 2000 or 3000 (bit better).

    """
    self.tokenizer = tokenizer
    
    self.sentence_num = -1
    self.clusters_num = 0

    self.token_map_context, self.annotation_context = [], []
    self.annotation_coreference_start, self.annotation_coreference_end = [], []
    self.token_map, self.annotation = [], []

    # a mention index to cluster mapping, e.g. (23, 24) -> [(23, 24), (41, 42)]
    self.mention_index_to_cluster = {}

    # the first link names the cluster, e.g. (23, 24) -> '1'
    self.mention_index_to_cluster_name = {}
    self.cluster_name_to_cluster = {}

    self.input_document = input_document
    # print('sentence_num', self.sentence_num)
    self.genre = input_document['genres'][0][0]
    self.speakers = {t: spk for (t, spk) in self.input_document['speakers']}

    self.done = False
    self.predictions_str = {}  # keep the predictions
    self.node_wise = node_wise

    self.max_len_doc = max_len_doc

    # move to initial position.
    self.extend()


  def extend_done(self):
    return self.done

  def extend(self, prediction_str=None, use_gold_cluster=False, move=True):

    # move annotation to context
    self.token_map_context +=  self.token_map
    self.annotation_context += self.annotation

    for k in range(len(self.annotation)):
      self.annotation_coreference_start.append([])
      self.annotation_coreference_end.append([])

    assert len(self.annotation_context)  == len(self.annotation_coreference_start)

    self.annotation, self.token_map = [], []

    link_found = False
    if prediction_str is not None and not 'None [' in prediction_str:
      links = [l for l in prediction_str.split(';;') if l != '' ]

      annotation_update = []
      for link in links:
        link_found = True
        link_mentions = get_mentions_for_link_state(link, self.node_wise)

        if len(link_mentions) < 2 and not (self.node_wise and len(link_mentions)):
          print('less mentions as needed skip', link_mentions)
          continue
        indices = match_link_state(link_mentions, self.annotation_full,
                                   self.annotation_full_map,
                                   self.cluster_name_to_cluster,
                                   debug=False)

        if not indices:
          print('not found !!')
          print('indices not found', link, indices)
          print('self.annotation_full', self.annotation_full )
          print('annotation + context', self.get_input_annotation())
          continue

        if True:
          index = indices[0]
          cluster = []
          for mention_index in index:
            if str(mention_index) in self.mention_index_to_cluster:
              cluster = self.mention_index_to_cluster[str(mention_index)]
              break


          if not cluster:
            self.clusters_num += 1
            cluster_name = str(self.clusters_num)

            if use_gold_cluster:  # just to evaluate on gold

              for ni, cx in enumerate(self.input_document['clusters']):
                for mx in cx:
                  if mx in index:
                    cluster_name = str(ni+1)
                    break

          else:
            cluster_name = self.mention_index_to_cluster_name[str(cluster[0])]

          for mention_index in index:
            if mention_index not in cluster:
              cluster.append(mention_index)
              self.mention_index_to_cluster[str(mention_index)] = cluster
              self.cluster_name_to_cluster['['+cluster_name] = cluster
              self.mention_index_to_cluster_name[str(mention_index)] = cluster_name
              annotation_update.append([mention_index, cluster_name])

      # update the annotation
      if True:
        for update in annotation_update:
          update_index = update[0]
          update_name = update[1]

          for t, coref_starts, coref_end, tid in zip(self.annotation_context,
                                    self.annotation_coreference_start,
                                    self.annotation_coreference_end,
                                    self.token_map_context):



            if update_index[0] == tid:
              coref_starts.append(update)
              coref_starts.sort( key=lambda x: x[0][1], reverse=True)


            if update_index[1] == tid:
              coref_end.append(']')

    if move or 'None [' in prediction_str or not link_found:
      self.sentence_num += 1

      if self.sentence_num not in self.input_document['sentences']:
        self.done = True
        return True

      tokens = self.input_document['sentences'][self.sentence_num]
      token_map = self.input_document['token_maps'][self.sentence_num]
      first = True


      for tid, t in zip(token_map, tokens):
        if first:
          self.token_map.append(-1)
          speaker = normalize_speaker(self.speakers[tid])
          self.annotation.append(speaker)
          first = False
        self.token_map.append(tid)
        self.annotation.append(t)

    if self.sentence_num not in self.predictions_str:
      self.predictions_str[self.sentence_num] = ''

    if prediction_str is not None:
      self.predictions_str[self.sentence_num] += prediction_str

    return False

  def input_annotation(self):

    self.annotation_full = ['coref:', self.genre]
    self.annotation_full_map = [-1, -1]
    for t, coref_starts, coref_end, tid in zip(self.annotation_context,
                                  self.annotation_coreference_start,
                                  self.annotation_coreference_end,
                                  self.token_map_context):

      for coref_start in coref_starts:
        coref_name = coref_start[-1]

        self.annotation_full.append('[' + coref_name)
        self.annotation_full_map.append(-1)

      self.annotation_full.append(t)
      self.annotation_full_map.append(tid)

      for end in coref_end:
        coref_name = end[-1]
        self.annotation_full.append(coref_name)
        self.annotation_full_map.append(-1)

    self.annotation_full += ['|'] + self.annotation
    self.annotation_full_map += [-1] + self.token_map
    self.annotation_full += ['**']
    self.annotation_full_map += [-1]


  def encode(self, annotation_str):
    # return tokenizer_mt5.encode(annotation_str)
    return self.tokenizer(annotation_str, add_special_tokens=False).input_ids

  def get_input_annotation(self, context_right=True):

    self.input_annotation()
    annotation_str = ' '.join(self.annotation_full)

    enc = self.encode(annotation_str)
    shorten = len(enc) > self.max_len_doc

    while len(enc) > self.max_len_doc:   # inefficient ...
      self.annotation_context = self.annotation_context[1:]
      self.token_map_context = self.token_map_context[1:]
      self.annotation_coreference_start = self.annotation_coreference_start[1:]
      self.annotation_coreference_end = self.annotation_coreference_end[1:]

      self.input_annotation()
      annotation_str = ' '.join(self.annotation_full)
      enc = self.encode(annotation_str)

    last_token_id = self.annotation_full_map[-2]  # the last one is **
    self.annotation_context_right = []

    if not shorten and context_right:
      sentence_num = self.sentence_num
      total_len = len(enc)

      while True:
        sentence_num += 1
        if sentence_num not in self.input_document['sentences']:
          break

        first = True
        annotation_context_next = []

        for t, tid in zip(self.input_document['sentences'][sentence_num], self.input_document['token_maps'][sentence_num]):
          if first:
            speaker = normalize_speaker(self.speakers[tid])
            annotation_context_next.append(speaker)
            first = False
          annotation_context_next.append(t)

        annotation_context_right = self.annotation_context_right + annotation_context_next
        enc = self.encode(' '.join(annotation_context_right))

        if (len(enc) + total_len) > self.max_len_doc:
          break
        self.annotation_context_right = annotation_context_right
      if self.annotation_context_right:
        annotation_str = annotation_str + ' ' + ' '.join(self.annotation_context_right)

    enc = self.encode(annotation_str)
    if len(enc) > self.max_len_doc:
      print('warning: document too long', len(enc))

    return annotation_str
