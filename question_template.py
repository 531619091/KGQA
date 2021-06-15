#-*- coding: UTF-8 -*-

from query import Query
import re

class QuestionTemplate():
    def __init__(self):
        self.q_template_dict={
            0:self.get_holder_by_stock, #ORG的股东有哪些
            1:self.get_stock_by_holder,#ORG持有哪些公司的股票
            2:self.get_org_per_stayed,#PER在过的机构
            3:self.get_company_per_served,#PER任职过的公司
            4:self.get_manager_by_company,#ORG的高管
            5:self.get_per_by_org,#在ORG学习或工作过的人
            6:self.get_paths_of_nodes,#返回空就行
            7:self.get_indicator,# 返回空就行
            8:self.get_indicator,# 返回空就行
            9:self.get_indicator,# 返回空就行
            10:self.get_indicator  # 返回空就行

        }

        # 连接数据库
        self.graph = Query()
        # 测试数据库是否连接上
        # result=self.graph.run("match (m:Movie)-[]->() where m.title='卧虎藏龙' return m.rating")
        # print(result)
        # exit()

    # def get_question_answer(self,question,template,question_entity):
    #     # 如果问题模板的格式不正确则结束
    #     assert len(str(template).strip().split("\t"))==2
    #     template_id,template_str=int(str(template).strip().split("\t")[0]),str(template).strip().split("\t")[1]
    #     self.template_id=template_id
    #     self.template_str2list=str(template_str).split()
    #
    #     # 预处理问题
    #     question_word,question_flag=[],[]
    #     for one in question:
    #         word, flag = one.split("/")
    #         question_word.append(str(word).strip())
    #         question_flag.append(str(flag).strip())
    #     assert len(question_flag)==len(question_word)
    #     self.question_word=question_word
    #     self.question_flag=question_flag
    #     self.raw_question=question
    #     # 根据问题模板来做对应的处理，获取答案
    #     answer=self.q_template_dict[template_id]()
    #     return answer

    def get_question_answer(self, template, question_entity):
        self.question_entity = question_entity
        print('template_id:', template)
        answer = self.q_template_dict[template]()
        return answer


    # 获取控制人
    # def get_controller(self):
    #     ## 获取ncontrol在原问题中的下标
    #     tag_index = self.question_flag.index("ncontrol")
    #     ## 获取控制人名称
    #     controller_name = self.question_word[tag_index]
    #     return controller_name
    #
    # # 获取公司名字
    # def get_company(self):
    #     ## 获取nstock在原问题中的下标
    #     tag_index = self.question_flag.index("nstock")
    #     ## 获取公司名称
    #     company_name = self.question_word[tag_index]
    #     return company_name
    #
    # # 获取股东名称
    # def get_holder(self):
    #     ## 获取nholder在原问题中的下标
    #     tag_index = self.question_flag.index("nholder")
    #     ## 获取股东名称
    #     holder_name = self.question_word[tag_index]
    #     return holder_name

    def format_answer(self,answer,concat_str):
        res = ''
        for key, values in answer.items():

            res += str(key) + concat_str + '</br>'

            for val in values:
                res += val + '</br>'
        return res

    # 获取公司股东
    def get_holder_by_stock(self):

        cql = f"match data=(n)-[:HOLD*]->(m) where (n:PERSON or n:COMPANY or n:SHAREHOLDER or n:ORGANIZATION) and m.name in {self.question_entity['KG_ORG']+self.question_entity['ORG']+self.question_entity['PER']} return m.name,n.name"
        cql_chart = f"match data=(n)-[:HOLD*]->(m) where (n:PERSON or n:COMPANY or n:SHAREHOLDER or n:ORGANIZATION) and m.name in {self.question_entity['KG_ORG']+self.question_entity['ORG']+self.question_entity['PER']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer',answer)

        print_answer = self.format_answer(answer,'的股东有：')
        print('final_answer', print_answer)
        return print_answer,answer,cql_chart

    # 获取股东持有的股票
    def get_stock_by_holder(self):

        cql = f"match (m)-[:HOLD*]->(n:COMPANY) where m.name in {self.question_entity['KG_ORG']+self.question_entity['PER']+self.question_entity['ORG']} return m.name,n.name"
        cql_chart = f"match data =(m)-[:HOLD*]->(n:COMPANY) where m.name in {self.question_entity['KG_ORG']+self.question_entity['PER']+self.question_entity['ORG']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer', answer)

        print_answer = self.format_answer(answer, '持有的股票有：')
        print('final_answer', print_answer)
        return print_answer, answer,cql_chart

    # PERSON在过的机构
    def get_org_per_stayed(self):
        #controller = self.get_controller()
        cql = f"match (m:PERSON)-[r:STAYED_IN]->(n) where m.name in {self.question_entity['PER']} return m.name,n.name"
        cql_chart = f"match data=(m:PERSON)-[r:STAYED_IN]->(n) where m.name in {self.question_entity['PER']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer', answer)

        print_answer = self.format_answer(answer, '待过的机构有：')
        print('final_answer', print_answer)
        return print_answer, answer,cql_chart

    # PERSON任职过的公司
    def get_company_per_served(self):
        #holder = self.get_holder()
        cql = f"match (m:PERSON)-[r:SERVED_IN]->(n) where m.name in {self.question_entity['PER']} return m.name,n.name"
        cql_chart = f"match data=(m:PERSON)-[r:SERVED_IN]->(n) where m.name in {self.question_entity['PER']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer', answer)

        print_answer = self.format_answer(answer, '任职过的上市公司有：')
        print('final_answer', print_answer)
        return print_answer, answer,cql_chart

    # 公司的高管
    def get_manager_by_company(self):
        #company_name = self.get_company()
        cql = f"match (m:PERSON)-[r:SERVED_IN]->(n:COMPANY) where r.on_job=\'1\' and n.name in {self.question_entity['KG_ORG']} return n.name,m.name"
        cql_chart = f"match data=(m:PERSON)-[r:SERVED_IN]->(n:COMPANY) where r.on_job=\'1\' and n.name in {self.question_entity['KG_ORG']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer', answer)

        print_answer = self.format_answer(answer, '的高管团队包括：')
        print('final_answer', print_answer)
        return print_answer, answer,cql_chart

    # 与某个机构有关的人
    def get_per_by_org(self):
        #company_name = self.get_company()
        cql = f"match (m:PERSON)-[r:STAYED_IN]->(n) where n.name in {self.question_entity['ORG']} return n.name,m.name"
        cql_chart = f"match data=(m:PERSON)-[r:STAYED_IN]->(n) where n.name in {self.question_entity['ORG']} return *"
        print(cql)
        answer = self.graph.run(cql)
        print('answer', answer)

        print_answer = self.format_answer(answer, ',在该机构学习或工作过的人有：')
        print('final_answer', print_answer)
        return print_answer, answer,cql_chart

    def get_paths_of_nodes(self):

        entity_list = self.question_entity['ORG'] + self.question_entity['PER']+self.question_entity['KG_ORG']
        cql_chart = f"MATCH (n:COMPANY) where n.name IN {entity_list} WITH collect(n) as nodes UNWIND nodes as n UNWIND nodes as m WITH * WHERE id(n) < id(m) MATCH path = allShortestPaths( (n)-[*..4]-(m) ) RETURN path"

        return 'ok', 'ok',cql_chart

    def get_indicator(self):
        return 'ok','ok','ok'

