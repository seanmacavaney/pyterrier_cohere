# pyterrier_cohere

PyTerrier bindings for Cohere Rerank

*(and maybe other stuff in the future)*

## Example

Example of Cohere ReRank:

```python
import pyterrier as pt
from pyterrier_cohere import Rerank
from pyterrier_pisa import PisaIndex
pt.init()

dataset = pt.get_dataset('irds:msmarco-passage')

bm25 = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=100)
rerank = Rerank('rerank-english-v3.0', api_key='your_api_key') # or from CO_API_KEY
pipeline = bm25 >> pt.text.get_text(dataset, 'text') >> rerank

pipeline.search('what does cohere mean?')
#   qid                   query    docno     score                                               text  rank
#     1  what does cohere mean?  2965451  0.999866  cohere (third-person singular simple present  ...     0
#     1  what does cohere mean?  1928828  0.999861  cohere (third-person singular simple present  ...     1
#     1  what does cohere mean?  1928833  0.999859  cohere (third-person singular simple present  ...     2
#     1  what does cohere mean?  1928832  0.999840  Definition of 'cohere'. cohere (koʊhɪəʳ ) If t...     3
#     1  what does cohere mean?  2965452  0.999743  Definition of cohere. cohered. ; cohering. int...     4
#    ..                     ...      ...       ...                                                ...   ...
```
