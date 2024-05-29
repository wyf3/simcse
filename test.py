from transformers import BertModel, BertConfig
from transformers import BertTokenizer

import numpy as np
from scipy.spatial.distance import cosine
def normalize(x):
    return x / np.linalg.norm(x)


def cosine_similarity(x, y):
    return 1 - cosine(x, y)

model = BertModel.from_pretrained("/home/wanyunfei/simcse/SimCSE/output/checkpoint-2000")
tokenizer = BertTokenizer.from_pretrained("/home/wanyunfei/simcse/SimCSE/output/checkpoint-2000")
model.eval()
s1 = tokenizer("一种分布式光储系统",return_tensors='pt')
s2 = tokenizer("一种分布式系统",return_tensors='pt')
s3 = tokenizer("一种洗涤剂系统",return_tensors='pt')

res1 = normalize((model(**s1).pooler_output).squeeze(0).detach().numpy())
res2 = normalize((model(**s2).pooler_output).squeeze(0).detach().numpy())
res3 = normalize((model(**s3).pooler_output).squeeze(0).detach().numpy())
print(cosine_similarity(res1, res2))
print(cosine_similarity(res1, res3))


#sentence_transformer加载

# from sentence_transformers import models,SentenceTransformer
# bert = models.Transformer('/home/wanyunfei/simcse/SimCSE/output/checkpoint-2000')
# pooler = models.Pooling(bert.get_word_embedding_dimension())
# normalize = models.Normalize()
# model = SentenceTransformer(modules=[bert, pooler, normalize])
# model.eval()
# s1 = model.encode("一种分布式光储系统")
# s2 = model.encode("一种分布式系统")
# s3 = model.encode("一种洗涤剂系统")
# print(cosine_similarity(s1, s2))
# print(cosine_similarity(s1, s3))