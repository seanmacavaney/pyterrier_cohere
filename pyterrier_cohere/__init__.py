import cohere
import more_itertools
import pyterrier as pt
import pandas as pd

class Rerank(pt.Transformer):
  def __init__(self, model, *, api_key=None, batch_size=100, verbose=True):
    self.model = model
    self.api = cohere.Client()
    self.batch_size = batch_size
    self.verbose = verbose

  def transform(self, inp):
    out = []
    it = inp.groupby('query')
    if self.verbose:
      it = pt.tqdm(it, unit='q', desc=repr(self))
    for query, df in it:
      new_scores = [None] * len(df)
      start_idx = 0
      for batch in more_itertools.chunked(df['text'], self.batch_size):
        res = self.api.rerank(model=self.model, query=query, documents=list(batch))
        for r in res.results:
          new_scores[start_idx + r.index] = r.relevance_score
        start_idx += self.batch_size
      df['score'] = new_scores
      out.append(df)
    out = pd.concat(out)
    pt.model.add_ranks(out)
    return out

  def __repr__(self):
    return f'Rerank({self.model!r})'

if __name__ == '__main__':
  if not pt.started():
    pt.init()
  from pyterrier_caching import ScorerCache

  dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
  rerank = Rerank('rerank-english-v3.0')
  scorer = ScorerCache('rerank-english-v3.0.msmarco-passage.cache', pt.text.get_text(dataset, 'text') >> rerank)
  if not scorer.built():
    scorer.build(dataset.get_corpus_iter())
  from pyterrier_pisa import PisaIndex
  from ir_measures import nDCG, R
  from pyterrier_adaptive import GAR, CorpusGraph
  graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_tcthnp_k16')
  bm25 = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=100)
  print(pt.Experiment(
    [
      bm25,
      bm25 >> scorer,
      bm25 >> GAR(scorer, graph, num_results=100),
    ],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG, R(rel=2)@100],
    names=['BM25', 'BM25 >> Rerank', 'BM25 >> GAR(Rerank)'],
    round=4,
  ))
