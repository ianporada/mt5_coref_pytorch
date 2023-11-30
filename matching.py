"""
Functions for matching mentions directly from the original notebook:
https://github.com/google-research/google-research/tree/master/coref_mt5#coreference-resolution-through-a-seq2seq-transition-based-system
"""

def match_mention_state(m, inputs, maps, position=None, debug=False, start_index=0):

  if '##' in m:
    index_num = m.index('##')
  else:
    if not m[0].startswith('['):
      print('get_chains::error ## not in split', m)
    index_num = len(m)

  if ']]' in inputs:
    end_index = inputs.index(']]')
  elif '**' in inputs:
    end_index = inputs.index('**')
  else:
    end_index = len(inputs)

  # m_clean = [x for x in m if x != '##']
  m_clean = []
  for x in m:
    if x != '##':
      m_clean.append(x)
    if x == '**':
      break

  # get context
  context = []
  found_num = False
  for s in m:
    if found_num:
      context.append(s)
    if '##' == s:
      found_num = True

  maps_index = 0
  indices = []
  for i in range(start_index, end_index):
    maps_index = i
    if inputs[i] == m_clean[0]:
      if inputs[i:i+len(m_clean)] == m_clean:
        indices.append((maps[maps_index], maps[maps_index + index_num  - 1]))

        if maps[maps_index + index_num  - 1] == -1:
          print('index negative', maps[maps_index:], ' index_num',  index_num)
          print('index negative', inputs[i:], ' index_num',  index_num)
          print(f'i {i} maps_index {maps_index}')


  if len(indices) == 0:
    print('none found match_mention', m)
    print('inputs', inputs)
    return []
  elif len(indices) > 1 and debug:
    print('match_mention: too many ', m,  indices, 'm_clean - use both')

  if (-1,-1) in indices:
    print('error for ',m, indices)
    return []

  return indices


def match_link_state(link, inputs, maps, cluster_name_to_cluster,
                     debug=True, node_wise=True):
  link_mentions = [m.split(' ') for m in link]
  links = []
  if len(link_mentions) == 1 and node_wise:
    m0 = link_mentions[0]
    try:
      index_m0 = match_mention_state(m0, inputs, maps, position=None)
      links = [index_m0]
    except Exception as e:
      print(str(e))
    return links


  m0 = link_mentions[0]
  m1 = link_mentions[1]

  if debug:
    print('match_link', m0, m1)

  # invert indices
  if m1[0].startswith('[') and len(m1[0]) > 0:
    cluster = cluster_name_to_cluster.get(m1[0], None)
    if cluster is not None:
      index_m1 = [cluster[-1]]
    else:
      print('cluster does not exists')
      return []
  else:
    index_m1 = match_mention_state(m1, inputs, maps, position=None)


  if debug:
    print(index_m1 ,'match' ,m1)

  if len(index_m1) > 1:
    print('index_m1', index_m1)

  try:
    index_m0 = match_mention_state(m0, inputs, maps, position=None)
  except Exception as e:
    print('error', str(e))
    index_m0 = []

  if debug:
    print(index_m0 ,'match' , m0)

  if len(index_m0) > 1:
    print('index_m0', index_m0)

  if len(index_m1) > 0 and len(index_m0) > 0:
      i1 = index_m1[-1]
      i2 = index_m0[-1]
      links.append([i1, i2])

  # use only last link
  if len(links) > 1:
    print('too many links, ', links, 'for link', link)
    print('context', inputs)

    return links[-1:]

  return links


def get_mentions_for_link_state(link, node_wise):
  link_split = link.split('->')

  if node_wise and len(link_split) == 1:
    m0 = link_split[0].strip()
    # print('link has only one mention?', link, m0)
    return [m0]

  elif len(link_split) < 2:
    print('link has only one mention - skipping mention', link)
    return []

  if len(link_split) > 2:
    print('link has too many mentions - using first two.', link)
  m0 = link_split[0].strip()
  m1 = link_split[1].strip()
  return [m0, m1]
