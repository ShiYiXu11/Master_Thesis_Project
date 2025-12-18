import pyspark
from pyspark.sql import SparkSession
# Import necessary functions
from pyspark.sql.functions import count as spark_count, desc, col, split, broadcast, lit, rand, concat, floor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, when
import networkx as nx
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
import networkx as nx
import pickle
from datetime import date
import numpy as np
from datetime import datetime
import KG_from_parquet as KGF
from vllm import LLM, SamplingParams
import networkx as nx
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import os
from graphframes import GraphFrame
from pyspark.sql import functions as F
import csv
import re 
import logging

# 1. 配置日志设置（放在代码最开头）
logging.basicConfig(
    filename='error_log.txt',  # 日志文件名
    level=logging.ERROR,       # 只记录 ERROR 及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s' # 日志格式：时间 - 级别 - 内容
)

rel_path = "../data/KG/rebel_relations.pkl"
en_map = "../data/KG/mapping.pkl"
entity_path = "../data/KG/rebel_entities.pkl"
style = "oral dialogue"
num_triplets =2

conf = (
    pyspark.SparkConf()
    .setAppName("WikipediaProcessing-SkewDiagnosis") # Changed AppName for clarity
    .setMaster("local[10]")  
    .setAll(
        [
            ("spark.driver.memory", "250g"),
            ("spark.driver.maxResultSize", "32G"), 
            ("spark.memory.fraction", "0.75"),
            ("spark.sql.shuffle.partitions", "2000"), # 8000 may be excessive for just groupBy
            ("spark.driver.memoryOverhead", "8g"),
            ("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12")
        ]
    )
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# initialize llm 
model_path = "meta-llama/Llama-3.2-1B-Instruct"
llm = LLM(model_path)
sampling_param = SamplingParams(
    temperature=0.0,  # 确保确定性
    max_tokens=50
)
# dics
rel_df, mapping_df, entity_df = KGF.load_dictionaries(spark,rel_path,en_map,entity_path)
# inputs
all_rows = spark.read.parquet("../data/KG/test_output/graph_data_all.parquet")
# graph
graph_simple_gf = KGF.create_graphframe_from_spark(
    spark, 
    all_rows, 
    mapping_df, 
    rel_df, 
    entity_df, 
    simple=True
)
# dic
with open(en_map, "rb") as f:
    id2name = pickle.load(f)
id2name_broadcast = spark.sparkContext.broadcast(id2name)

@F.udf("string")
def lookup_name(qid):
    return id2name_broadcast.value.get(qid, qid)

# 解析 relation_Fid 中的属性部分（Pxxx）
@F.udf("string")
def extract_property(rel_id):
    parts = rel_id.split('_')
    if len(parts) == 3:
        return parts[1]  # e.g. Q31_P361_Q13116 → P361
    return None

def choose_id_LLM(prop_pairs_set,subjects, llm, sampling_params, style):
    prompt = f"""You are an AI evaluator. Select 1 or 2 relations that are suitable to generate {style} for {subjects}. Only output the relations id which formatted as Pxx. Do NOT add extra information.
                Relations:
                {prop_pairs_set}
                id:"""
    print(prompt)
    outputs = llm.generate([prompt], sampling_params)
    x = outputs[0].outputs[0].text.strip()
    print(x)
    x=re.findall('P\d+',x)
    if len(x)>2:
        return x[:2]
    else:
        return list(x[0])
def choose_next_node(subjects, node_set, llm, sampling_params, style):
    # prompt = f"""You are an AI evaluator. Select only one next one node from nodes to continue generating news based on the subjects {subjects}. Output the keys as Qx.
                
    #             nodes:
    #             {node_set}
                
    #             output:"""
    prompt=f"""You are an AI evaluator. Your goal is to select one most relevant entity to continue generating {style} about the subject: '{subjects}'.

Select only one entity from the list below.

Entities:
{node_set}

Respond with *only* the ID of your choice, which formatted as Qxxx. Do not add any explanation or other text.

Output:"""
    print(prompt)
    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0])
    x = outputs[0].outputs[0].text.strip()
    print(x)
    x=re.findall('Q\d+',x)
    return x[0]
    # ids = [x.strip() for x in x.split(',') if x.strip()]
    # return ids[0]
def filter_df(graph_simple_gf,current_src ):
    same_src_df = graph_simple_gf.edges.filter(F.col("src")== current_src) # filter same src 
    if same_src_df.isEmpty():
        return None, None
    else:
        # concat tables on selected id
        edges_named = (
        same_src_df
        .withColumn("src_name", lookup_name("src"))
        .withColumn("dst_name", lookup_name("dst"))
        .withColumn("prop", extract_property("relation_id"))
        .withColumn("prop_name", lookup_name("prop"))
        .withColumn(
            "relation_name",
            F.concat_ws("", F.concat(F.lit("<subj>"), F.col("src_name")), 
                        F.concat(F.lit("<rel>"), F.col("prop_name")),
                        F.concat(F.lit("<obj>"),F.col("dst_name"))
        )
        )
        )
        # prop_pairs = edges_named.select("prop", "prop_name") \
        #         .rdd.map(lambda row: f"<id>{row['prop']}<name>{row['prop_name']}") \
        #         .collect()
        relations_dict = {
        row['prop']: row['prop_name'] 
        for row in edges_named.select("prop", "prop_name").collect()
    }
        return relations_dict,edges_named

with open(entity_path, "rb") as f:
    rebel_entities = pickle.load(f)
with open(f"relations_B_{style}_new.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relations"])  # 只写一列，每行一个 relation list

    # 取前100个 src 节点
    #src_nodes = list(set([row['src'] for row in graph_simple_gf.edges.select('src').limit(1000).collect()]))
    start_nodes = set(list(rebel_entities)[:200])
    # src_nodes= ['Q44578']
    for src in start_nodes:
        try:
            subjects_names=[]
            rel=set()
            for i in range(np.random.poisson(2)+1):
                relations_dict,edges_named = filter_df(graph_simple_gf,src)
                src_name = edges_named.select("src_name").first()["src_name"]
                subjects_names.append(src_name)
                next_rel_ids = choose_id_LLM(prop_pairs_set=relations_dict, subjects=subjects_names,llm=llm, sampling_params=sampling_param, style =style)
                if len(next_rel_ids) > 0: 
                        same_src_rel_def= edges_named.filter((F.col("prop").isin(next_rel_ids)))
                        string_list = [row['relation_name'] for row in same_src_rel_def.select("relation_name").collect()]
                        rel.update(string_list)
                        rows = same_src_rel_def.select("dst", "dst_name").collect()

                        dst_names_dict = {row['dst']: row['dst_name'] for row in rows}
                        # rel.extend(string_list)
                        # # dst = same_src_rel_def.select("dst").first()['dst']
                        next_src=choose_next_node(subjects=subjects_names, node_set=dst_names_dict, llm=llm, sampling_params=sampling_param, style = style)
                        src = next_src
                else:
                    writer.writerow([])
            writer.writerow([rel])
            print("final:"+str(rel))
        except Exception as e:
            logging.error("error", exc_info=True)
            writer.writerow([])
            continue
