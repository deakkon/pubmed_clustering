from utils.preprocess import preprocess_text
from utils.transformers import TokenizePreprocessor

pt = preprocess_text()
tp = TokenizePreprocessor()

print(pt.gold_data_labeled)
gold_data = pt.get_text(pt.gold_data_labeled.PMID.values.tolist(), 'gold_text',pt.gold_data_labeled.Label.values.tolist())
test_data = pt.get_text(pt.test_data.PMID.values.tolist(), 'test_data')


gold_text = [x['title']+x['abstract'] for x in gold_data]
tranformed_gold_data = tp.transform(gold_text)
print(tranformed_gold_data)
pt.transform_text(tranformed_gold_data, pt.gold_data_labeled.Label.values.tolist())
