import pyterrier as pt
from pyterrier_cohere import Rerank
from pyterrier_caching import ScorerCache
from pyterrier_adaptive import GAR, CorpusGraph
from pyterrier_pisa import PisaIndex
from ir_measures import nDCG, R

if not pt.started():
  pt.init()

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

# pipeline components
rerank = Rerank('rerank-english-v3.0')
bm25 = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=100)
graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_tcthnp_k16')

# avoid re-scoring documents for experiments
cached_rerank = ScorerCache('rerank-english-v3.0.msmarco-passage.cache', pt.text.get_text(dataset, 'text') >> rerank)
if not cached_rerank.built():
  cached_rerank.build(dataset.get_corpus_iter())

# BM25 vs BM25 >> Rerank vs BM25 >> GAR(Rerank)
print(pt.Experiment(
  [
    bm25,
    bm25 >> cached_rerank,
    bm25 >> GAR(cached_rerank, graph, num_results=100),
  ],
  dataset.get_topics(),
  dataset.get_qrels(),
  [nDCG@10, nDCG, R(rel=2)@100],
  names=['BM25', 'BM25 >> Rerank', 'BM25 >> GAR(Rerank)'],
  round=4,
))
