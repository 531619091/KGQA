# KGQA
这是一个知识图谱问答系统，使用的图数据库为neo4j
本系统用到的bert模型取自https://github.com/valuesimplex/FinBERT

该系统是一个检索型问答系统，系统以数据库的形式把上市公司相关信息进行存储，同时预先设置了一些问题模板，
当用户提问时，系统判断该问题对应哪个模板，然后使用该模板对应的查询语句在数据库进行查询，将查询结果返回给用户，一轮对话结束。
流程如下：

![image](https://user-images.githubusercontent.com/48402229/121984543-babafc80-cdc5-11eb-930b-9de75ca282dd.png)

对话流程中每个模块用到的技术如下：

![image](https://user-images.githubusercontent.com/48402229/121984924-6bc19700-cdc6-11eb-85e8-4140b7817926.png)

下图描述了每个程序文件的作用：

![image](https://user-images.githubusercontent.com/48402229/121984988-8562de80-cdc6-11eb-9dcf-541c43928f61.png)

下面演示一些问答结果：

查询公司股东

![image](https://user-images.githubusercontent.com/48402229/121985126-b93e0400-cdc6-11eb-93a5-7df2d5922844.png)

查询财务报表

![image](https://user-images.githubusercontent.com/48402229/121985132-bf33e500-cdc6-11eb-83fe-e85046aaeaf3.png)
