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


# --- Spark Configuration ---

rel_path = "rebel_relations.pkl"
en_map = "mapping.pkl"
entity_path = "rebel_entities.pkl"
# spark initialization
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
            ("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
        ]
    )
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# initialize llm 
model_path = "../models/Llama-2-13B-chat-hf"
llm = LLM(model_path)
sampling_params = SamplingParams(
    temperature=0.0,  # 确保确定性
    max_tokens=50
)
# dics
rel_df, mapping_df, entity_df = KGF.load_dictionaries(spark,rel_path,en_map,entity_path)
# inputs
all_rows = spark.read.parquet("./test_output/graph_data_all.parquet")
# graph
graph_simple_gf = KGF.create_graphframe_from_spark(
    spark, 
    all_rows, 
    mapping_df, 
    rel_df, 
    entity_df, 
    simple=True
)

# 字典映射转成 UDF
with open(en_map, "rb") as f:
    id2name = pickle.load(f)
id2name_broadcast = spark.sparkContext.broadcast(id2name)

@F.udf("string")
def lookup_name(qid):
    return id2name_broadcast.value.get(qid, qid)

# 解析 relation_id 中的属性部分（Pxxx）
@F.udf("string")
def extract_property(rel_id):
    parts = rel_id.split('_')
    if len(parts) == 3:
        return parts[1]  # e.g. Q31_P361_Q13116 → P361
    return None

def choose_id_LLM(prop_pairs_set, llm, sampling_params):
    prompt = f"""You are an AI evaluator. Select relations that are suitable to generate news for {src}. Only output the relations id. Do NOT add extra information.
Relations:
{prop_pairs_set}
id:"""
    outputs = llm.generate([prompt], sampling_params)
    x = outputs[0].outputs[0].text.strip()
    ids = [x.strip() for x in x.split(',') if x.strip()]
    return ids[0]

import csv
with open("relations.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relations"])  # 只写一列，每行一个 relation list

    # 取前100个 src 节点
    #src_nodes = list(set([row['src'] for row in graph_simple_gf.edges.select('src').limit(1000).collect()]))
    src_nodes= ['Q8','Q42']

    for src in src_nodes:
        num_triplets = np.random.poisson(lam=3.0)
        rel = []
        current_src = src
        for i in range(2, num_triplets + 1): # how many triples, poisson here
            # find start here
            same_src_df = graph_simple_gf.edges.filter(F.col("src") == current_src) # filter same src 
            if same_src_df.count() == 0:
                break
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
            start_name = edges_named.select("src_name").first()["src_name"]
            
            
            # 2️⃣ 收集 relation_name 列
            prop_pairs = edges_named.select("prop", "prop_name") \
                .rdd.map(lambda row: f"<id>{row['prop']}<name>{row['prop_name']}") \
                .collect()
            prop_pairs_set = set(prop_pairs)
            next_rel_id = choose_id_LLM(prop_pairs_set=prop_pairs_set, llm=llm, sampling_params=sampling_params)
        
        
            if next_rel_id  is not None: 
                try:
                    same_src_rel_def= edges_named.filter(F.lower(F.col("prop"))==next_rel_id.lower())
                    rows_list = same_src_rel_def.select("relation_name").collect()
                    string_list = [row['relation_name'] for row in rows_list]
                    rel.extend(string_list)
                    dst = same_src_rel_def.select("dst").first()['dst']
                    current_src = dst
                except Exception as e:
                    print(e)
                    rel.append("")
            else:
                rel.append("none")
                break
        writer.writerow([rel])
        print(rel)