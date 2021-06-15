#-*- coding: UTF-8 -*-

'''
接收原始问题
对原始问题进行分词、词性标注等处理
对问题进行抽象
'''

import jieba.posseg
import re

from question_template import QuestionTemplate
from LAC import LAC
from collections import defaultdict
import torch
from transformers import BertTokenizer,BertModel,BertConfig
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Timeline, Grid, Bar, Map, Pie, Line,Graph,Tab
from fuzzywuzzy import fuzz
import numpy as np
from fuzzywuzzy import process
import pandas as pd
from jqdatasdk import *
auth('输入账户','输入密码')
# # 将自定义字典写入文件
# result = []
# with(open("./data/userdict.txt","r",encoding="utf-8")) as fr:
#     vocablist=fr.readlines()
#     for one in vocablist:
#         if str(one).strip()!="":
#             temp=str(one).strip()+" "+str(15)+" nr"+"\n"
#             result.append(temp)
# with(open("./data/userdict2.txt","w",encoding="utf-8")) as fw:
#     for one in result:
#         fw.write(one)

import sys, os
import pickle

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
# blockPrint()

# enablePrint()



class Question():
    def __init__(self):
        self.init_config()

    def init_config(self):

        # 创建问题模板对象
        self.questiontemplate=QuestionTemplate()

        model_dir = "./bert"
        VOCAB = "vocab.txt"
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, VOCAB))
        self.model = BertModel.from_pretrained(model_dir, output_hidden_states=True)  # 如果想要获取到各个隐层值需要如此设置
        self.model.eval()
        template0 = ['KG_ORG的股东有哪些','哪些公司入股了KG_ORG','KG_ORG的大股东','KG_ORG股东','KG_ORG的持股人']
        template1 = ['KG_ORG持有哪些公司的股票','KG_ORG入股了哪些公司','KG_ORG持股公司','KG_ORG入股的公司']
        template2 = ['PER在过的机构','PER简历','PER经历','PER工作学历经历']
        template3 = ['PER任职过的公司','PER在哪些公司当过高管','PER上市公司任职经历','PER服务过的公司','PER管理过哪些公司']
        template4 = ['KG_ORG的高管有哪些','KG_ORG管理层','KG_ORG管理人员','KG_ORG管理团队','管理KG_ORG的人员']
        template5 = ['在ORG学习或工作过的人','ORG出来的人','在ORG待过的人']
        template6 = ['KG_ORG和KG_ORG之间有什么关系','ORG和ORG之间的路劲','ORG和ORG关联','ORG和ORG关系','ORG和ORG的共同点','这些ORG间的共性','查看关系']
        template7 = ['查看KG_ORG的利润表','KG_ORG利润']
        template8 = ['查看KG_ORG的资产负债表','KG_ORG资产']
        template9 = ['查看KG_ORG的现金流量表','KG_ORG现金流']
        with open('./data/indicator_mean_vec','rb') as f:
            indictor_mean_vec = pickle.load(f)
        self.template = [template0,template1,template2,template3,template4,template5,template6,template7,template8,template9]
        self.template_embedding = [self.sentence_mean_vec(temp) for temp in self.template]+[indictor_mean_vec]

        with open('./data/indicator_dict','rb') as f:
            self.indicator_dict = pickle.load(f)
        with open('./data/result_data','rb') as f:
            self.result_data = pickle.load(f)
        with open('./data/kg_dict','rb') as f:
            self.kg_dict = pickle.load(f)


    def sentence_mean_vec(self,sentence_list):
        list_vec = [self.get_sentence_vector(sentence).numpy() for sentence in sentence_list]
        result = torch.from_numpy(np.mean(list_vec,axis=0))
        return result

    def get_sentence_vector(self,sentence):
        with torch.no_grad():
            sentence = "[CLS]" + sentence + "[SEP]"
            # Convert token to vocabulary indices
            tokenized_string = self.tokenizer.tokenize(sentence)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokenized_string)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([tokens_ids])
            outputs = self.model(tokens_tensor)  # encoded_layers, pooled_output

            if self.model.config.output_hidden_states:
                hidden_states = outputs[2]
                # last_layer = outputs[-1]
                second_to_last_layer = hidden_states[-2]
                # 由于只要一个句子，所以尺寸为[1, 10, 768]
                token_vecs = second_to_last_layer[0]

                sentence_embedding = torch.mean(token_vecs, dim=0)
                return sentence_embedding

    def get_question_template(self):
        cos_similarity = [torch.cosine_similarity(self.get_sentence_vector(self.processed_quesiton), embed, dim=0).item() for embed in
                          self.template_embedding]
        most_similar_index = cos_similarity.index(max(cos_similarity))
        return most_similar_index

    def question_process(self,question,flag='msg'):
        # 接收问题
        self.raw_question=str(question).strip()
        # 对问题进行标注
        self.processed_quesiton=self.question_posseg()
        # 得到问题的模板
        self.question_template_id=self.get_question_template()
        # 查询图数据库,得到答案
        self.print_answer,self.answer_dict,self.cql=self.query_template()
        self.chart = 'ok'
        if flag =='chart':
            self.chart = self.get_chart()
        if self.chart is None:
            self.chart = 'ok'
        if self.print_answer is None:
            self.print_answer = 'ok'
        return self.print_answer,self.chart,self.question_template_id,self.cql


    def question_posseg(self):
        lac = LAC(mode='lac')
        lac.load_customization('./data/custom.txt', sep=None)
        clean_question = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+","",self.raw_question)
        self.clean_question=clean_question
        lac_result = lac.run(self.clean_question)
        self.question_entity = defaultdict(list)
        sentence_process = ''
        for word, tag in zip(*lac_result):
            if tag in ['PER', 'ORG', 'LOC','KG_ORG']:
                w = tag
                if tag == 'KG_ORG':
                    self.question_entity[tag].append(process.extractOne(word, self.kg_dict)[0])
                else:
                    self.question_entity[tag].append(word)
                    print('****************',self.question_entity)
            else:
                w = word
            sentence_process += w
        print('sentence',sentence_process)
        return sentence_process


    # 根据问题模板的具体类容，构造cql语句，并查询
    def query_template(self):
        # 调用问题模板类中的获取答案的方法
        # try:
        #     answer=self.questiontemplate.get_question_answer(self.pos_quesiton,self.question_template_id_str)
        # except:
        #     answer="我也还不知道！"
        print('question_template_id',self.question_template_id)
        answer = self.questiontemplate.get_question_answer(self.question_template_id,self.question_entity)
        return answer

    def graph(self,nodes, links, categories) -> Graph:

        c = (Graph()
             .add("", nodes, links, categories, repulsion=4000, edge_label='持股')
             .set_global_opts(title_opts=opts.TitleOpts(title="网络图"))
             )

        return c

    def get_map_chart(self,question):
        col = process.extractOne(question, self.indicator_dict)[0]
        self.print_answer = col

        data = self.result_data.loc[:, ['地区', col]]
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data['percent'] = data[col] / data[col].sum()
        data = data.sort_values(by='percent', ascending=False)
        data_tuple = []
        for i, row in data.iterrows():
            data_tuple.append((row['地区'], row[col]))
        max_num, min_num = data[col].max(), data[col].min()
        map_chart = (
            Map()
                .add(
                series_name="",
                data_pair=data_tuple,
                zoom=1,
                center=[119.5, 34.5],
                is_map_symbol_show=False,
                layout_center=['70%', '40%'],
                layout_size='800',
                itemstyle_opts={
                    "normal": {"areaColor": "#323c48", "borderColor": "#404a59"},
                    "emphasis": {
                        "label": {"show": Timeline},
                        "areaColor": "rgba(255,255,255, 0.5)",
                    },
                },
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=col
                ),
                tooltip_opts=opts.TooltipOpts(
                    is_show=True,
                    formatter="{b}<br/>{c}",
                ),
                visualmap_opts=opts.VisualMapOpts(
                    is_calculable=True,
                    dimension=0,
                    pos_left="30",
                    pos_top="center",
                    range_text=["High", "Low"],
                    range_color=["lightskyblue", "yellow", "orangered"],
                    textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                    min_=min_num,
                    max_=max_num,
                ),
            )
        )

        bar_x_data = [x[0] for x in data_tuple]
        bar_y_data = [{"name": x[0], "value": x[1]} for x in data_tuple]
        bar_chart = (
            Bar()
                .add_xaxis(xaxis_data=bar_x_data)
                .add_yaxis(
                series_name="",
                y_axis=bar_y_data,
                label_opts=opts.LabelOpts(
                    is_show=True, position="right", formatter="{b} : {c}"
                ),
            )
                .reversal_axis()
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    max_=max_num, axislabel_opts=opts.LabelOpts(is_show=False)
                ),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
                tooltip_opts=opts.TooltipOpts(is_show=False),
                visualmap_opts=opts.VisualMapOpts(
                    is_calculable=True,
                    dimension=0,
                    pos_left="10",
                    pos_top="top",
                    range_text=["High", "Low"],
                    range_color=["lightskyblue", "yellow", "orangered"],
                    textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                    min_=min_num,
                    max_=max_num,
                ),
            )
        )

        pie_data = []

        for i, row in data.iterrows():
            pie_data.append([row['地区'], row['percent']])
        pie_chart = (
            Pie()
                .add(
                series_name="",
                data_pair=data_tuple,
                radius=["10%", "25%"],
                center=["85%", "85%"],
                itemstyle_opts=opts.ItemStyleOpts(
                    border_width=1, border_color="rgba(0,0,0,0.3)"
                ),
            )
                .set_global_opts(
                tooltip_opts=opts.TooltipOpts(is_show=True, formatter="{b} {d}%"),
                legend_opts=opts.LegendOpts(is_show=False),
                visualmap_opts=opts.VisualMapOpts(
                    is_calculable=True,
                    dimension=0,
                    pos_left="10",
                    pos_top="top",
                    range_text=["High", "Low"],
                    range_color=["lightskyblue", "yellow", "orangered"],
                    textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                    min_=min_num,
                    max_=max_num,
                ),
            )
        )

        grid_chart = (
            Grid(init_opts=opts.InitOpts(width="1200px", height="850px", theme=ThemeType.DARK))
                .add(
                bar_chart,
                grid_opts=opts.GridOpts(
                    pos_left=10, pos_right="45%", pos_top="50%", pos_bottom='5'
                ),
            )
                .add(pie_chart, grid_opts=opts.GridOpts(pos_left="85%", pos_bottom="40%", pos_right='5'))
                .add(map_chart, grid_opts=opts.GridOpts(pos_left='50', pos_right='40%'))
        )

        return grid_chart.dump_options_with_quotes()

    def get_balance_report(self):
        company_name = self.question_entity['KG_ORG'][0]
        full_name = finance.run_query(
            query(finance.STK_COMPANY_INFO).filter(finance.STK_COMPANY_INFO.short_name == company_name).limit(1))[
            'full_name'].values[0]

        balance_df = finance.run_query(query(finance.STK_BALANCE_SHEET) \
                                       .filter(finance.STK_BALANCE_SHEET.company_name == full_name, \
                                               finance.STK_BALANCE_SHEET.report_type == 0, \
                                               finance.STK_BALANCE_SHEET.report_date.ilike('_____12-31'))) \
            [['report_date', 'total_assets', 'total_current_assets', 'total_non_current_assets',
              'total_liability', 'total_current_liability', 'total_non_current_liability', 'total_owner_equities']]
        balance_df['year'] = balance_df['report_date'].apply(lambda x: str(x.year))
        total_assets = list(round(balance_df['total_assets'] / 100000000, 2))
        total_current_assets = list(round(balance_df['total_current_assets'] / 100000000, 2))
        total_non_current_assets = list(round(balance_df['total_non_current_assets'] / 100000000, 2))
        total_liability = list(round(balance_df['total_liability'] / 100000000, 2))
        total_current_liability = list(round(balance_df['total_current_liability'] / 100000000, 2))
        total_non_current_liability = list(round(balance_df['total_non_current_liability'] / 100000000, 2))
        total_owner_equities = list(round(balance_df['total_owner_equities'] / 100000000, 2))
        balance_chart = (
            Bar()
                .add_xaxis(balance_df['year'].to_list())
                .add_yaxis("总资产", total_assets, gap="0%")
                .add_yaxis("流动资产", total_current_assets, gap="0%")
                .add_yaxis("非流动资产", total_non_current_assets, gap="0%")
                .add_yaxis("总负债", total_liability, gap="0%")
                .add_yaxis("流动负债", total_current_liability, gap="0%")
                .add_yaxis("非流动负债", total_non_current_liability, gap="0%")
                .add_yaxis("股东权益", total_owner_equities, gap="0%")
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(title_opts=opts.TitleOpts(title="资产负债表"),
                datazoom_opts=opts.DataZoomOpts(),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}亿")), )
        )
        return balance_chart.dump_options_with_quotes()

    def get_cashflow_report(self):
        company_name = self.question_entity['KG_ORG'][0]
        full_name = finance.run_query(query(finance.STK_COMPANY_INFO).filter(finance.STK_COMPANY_INFO.short_name==company_name).limit(1))['full_name'].values[0]
        cashflow_df = finance.run_query(query(finance.STK_CASHFLOW_STATEMENT) \
                                        .filter(finance.STK_CASHFLOW_STATEMENT.company_name == full_name, \
                                                finance.STK_CASHFLOW_STATEMENT.report_type == 0, \
                                                finance.STK_CASHFLOW_STATEMENT.report_date.ilike('_____12-31'))) \
            [['report_date', 'net_operate_cash_flow', 'net_invest_cash_flow', 'net_finance_cash_flow',
              'cash_equivalent_increase', 'cash_and_equivalents_at_end']]
        cashflow_df['year'] = cashflow_df['report_date'].apply(lambda x: str(x.year))
        net_operate_cash_flow = list(round(cashflow_df['net_operate_cash_flow'] / 100000000, 2))
        net_invest_cash_flow = list(round(cashflow_df['net_invest_cash_flow'] / 100000000, 2))
        net_finance_cash_flow = list(round(cashflow_df['net_finance_cash_flow'] / 100000000, 2))
        cash_equivalent_increase = list(round(cashflow_df['cash_equivalent_increase'] / 100000000, 2))
        cash_and_equivalents_at_end = list(round(cashflow_df['cash_and_equivalents_at_end'] / 100000000, 2))

        line = (
            Line()
                .add_xaxis(xaxis_data=cashflow_df['year'].tolist())
                .add_yaxis(
                series_name="现金及现金等价物净增加额",
                y_axis=cash_equivalent_increase,
                label_opts=opts.LabelOpts(is_show=False),
            )
                .add_yaxis(
                series_name="年末现金及现金等价物",
                y_axis=cash_and_equivalents_at_end,
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        cashflow_bar = (
            Bar()
                .add_xaxis(cashflow_df['year'].tolist())
                .add_yaxis("经营活动产生的现金流量净额经营", net_operate_cash_flow, gap="0%")
                .add_yaxis("投资活动产生的现金流量净额", net_invest_cash_flow, gap="0%")
                .add_yaxis("筹资活动产生的现金流量净额", net_finance_cash_flow, gap="0%")
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(title_opts=opts.TitleOpts(title="现金流量表"),
                                 datazoom_opts=opts.DataZoomOpts(),
                                 yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}亿")), )
        )
        cashflow_chart = cashflow_bar.overlap(line)
        return cashflow_chart.dump_options_with_quotes()

    def get_profit_report(self):

        company_name = self.question_entity['KG_ORG'][0]
        full_name = finance.run_query(
            query(finance.STK_COMPANY_INFO).filter(finance.STK_COMPANY_INFO.short_name == company_name).limit(1))[
            'full_name'].values[0]
        profit_df = finance.run_query(query(finance.STK_INCOME_STATEMENT) \
                                      .filter(finance.STK_INCOME_STATEMENT.company_name == full_name, \
                                              finance.STK_INCOME_STATEMENT.report_type == 0, \
                                              finance.STK_INCOME_STATEMENT.report_date.ilike('_____12-31'))) \
            [['report_date', 'total_operating_revenue', 'total_operating_cost', 'operating_profit', 'total_profit',
              'income_tax', 'net_profit']]
        profit_df['year'] = profit_df['report_date'].apply(lambda x: str(x.year))
        total_operating_revenue = list(round(profit_df['total_operating_revenue'] / 100000000, 2))
        total_operating_cost = list(round(profit_df['total_operating_cost'] / 100000000, 2))
        operating_profit = list(round(profit_df['operating_profit'] / 100000000, 2))
        total_profit = list(round(profit_df['total_profit'] / 100000000, 2))
        income_tax = list(round(profit_df['income_tax'] / 100000000, 2))
        net_profit = list(round(profit_df['net_profit'] / 100000000, 2))


        profit_chart = (
            Bar()
                .add_xaxis(profit_df['year'].to_list())
                .add_yaxis("营业总收入", total_operating_revenue, gap="0%")
                .add_yaxis("营业总成本", total_operating_cost, gap="0%")
                .add_yaxis("营业利润", operating_profit, gap="0%")
                .add_yaxis("总利润", total_profit, gap="0%")
                .add_yaxis("净利润", net_profit, gap="0%")
                .add_yaxis("所得税", income_tax, gap="0%")
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(title_opts=opts.TitleOpts(title="利润表"),
                datazoom_opts=opts.DataZoomOpts(),
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}亿")), )
        )

        return profit_chart.dump_options_with_quotes()

    def get_chart(self):

        if self.question_template_id == len(self.template_embedding)-1:
            return self.get_map_chart(self.raw_question)
        elif self.question_template_id == len(self.template_embedding)-2:
            return self.get_cashflow_report()
        elif self.question_template_id == len(self.template_embedding)-3:
            return self.get_balance_report()
        elif self.question_template_id == len(self.template_embedding)-4:
            return self.get_profit_report()
        else:
            return 'ok'
            # if self.answer_dict:
            #     nodes = []
            #     links = []
            #     categories = []
            #     categories.append(opts.GraphCategory(name='0'))
            #     categories.append(opts.GraphCategory(name='1'))
            #     for key, values in self.answer_dict.items():
            #
            #         nodes.append(opts.GraphNode(name=key, symbol_size=50, category=0))
            #         for val in values:
            #             nodes.append(opts.GraphNode(name=val, symbol_size=50, category=1))
            #             links.append(opts.GraphLink(source=key, target=val))
            #
            #         c = self.graph(nodes, links, categories)
            #         return c.dump_options_with_quotes()

        return 'ok'




