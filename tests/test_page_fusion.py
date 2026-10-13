from rag import FinancialRAG

def test_page_fusion_deduplicates_and_returns_best_dense_excerpt(monkeypatch):
    monkeypatch.setenv('PAGE_DENSE_WEIGHT','3')
    monkeypatch.setenv('PAGE_LEXICAL_WEIGHT','1')
    r=FinancialRAG()
    r.chunks=[
      {'doc':'a','page':1,'text':'dense-first'},
      {'doc':'a','page':1,'text':'dense-duplicate'},
      {'doc':'a','page':2,'text':'lexical'},
      {'doc':'a','page':3,'text':'other'},
    ]
    result=r._rank_pages('arbitrary', [0,1,3], [2,1,0],2)
    assert len(result)==2
    assert len({(x['doc'],x['page']) for x in result})==2
    assert result[0]['text']=='dense-first'

def test_page_fusion_rejects_bad_weight(monkeypatch):
    monkeypatch.setenv('PAGE_DENSE_WEIGHT','-1')
    r=FinancialRAG();r.chunks=[{'doc':'a','page':1,'text':'x'}]
    try:
        r._rank_pages('x',[0],[0],1)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected invalid weight to fail')
