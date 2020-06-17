> random_embedding

随机初始化embedding

> transformers

BertsForTokenClassification -> bertsModel + softmax
Model | Acc | Recall| F1| Training Time | epoch | lr
--- | --- | --- | --- | --- | --- | ---
Bert_base|0.947112462006079|0.9635126777983921|0.955242182709994|494 | 10 |5e-5
DistillBert_base|0.9254093389933293|0.9437229437229437|0.934476423759951|927 |15 | 5e-5
Roberta_base|0.8434430964760254|0.9029066171923315|0.8721624850657109|5226 |50 | 5e-5
Albert_base|0.7311422413793104|0.839208410636982|0.7814569536423841|4633 |60 | 5e-5

