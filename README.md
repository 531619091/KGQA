# KGQA
这是一个知识图谱问答系统，使用的图数据库为neo4j
本系统用到的bert模型取自https://github.com/valuesimplex/FinBERT

该系统是一个检索型问答系统，系统以数据库的形式把上市公司相关信息进行存储，同时预先设置了一些问题模板，
当用户提问时，系统判断该问题对应哪个模板，然后使用该模板对应的查询语句在数据库进行查询，将查询结果返回给用户，一轮对话结束。
![image](https://user-images.githubusercontent.com/48402229/121984543-babafc80-cdc5-11eb-930b-9de75ca282dd.png)
