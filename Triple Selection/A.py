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
import random
from pyspark.sql import functions as F
import csv
os.environ["HF_HOME"] = "/work/shiyxu/hf_cache"
os.environ["HF_TOKEN"] = "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"

style ="twitter"
rel_path = "../data/KG/rebel_relations.pkl"
en_map = "../data/KG/mapping.pkl"
entity_path = "../data/KG/rebel_entities.pkl"

conf = (
    pyspark.SparkConf()
    .setAppName("WikipediaProcessing-SkewDiagnosis") # Changed AppName for clarity
    .setMaster("local[10]")  
    .setAll(
        [
            ("spark.driver.memory", "250g"),
            ("spark.driver.maxResultSize", "32G"), 
            ("spark.memory.fraction", "0.85"),
            ("spark.sql.shuffle.partitions", "2000"), # 8000 may be excessive for just groupBy
            ("spark.driver.memoryOverhead", "8g"),
            ("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12")
        ]
    )
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# initialize llm 
model_path = "meta-llama/Llama-3.2-1B-Instruct"
llm = LLM(model=model_path, download_dir="/work/shiyxu/hf_cache")
sampling_params = SamplingParams(
    temperature=0.0,
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
graph_simple_gf.edges.show()

import re
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
        start_name = edges_named.select("src_name").first()["src_name"]
        edges_named.show()
        relation_name_set = {row['relation_name'] for row in edges_named.select("relation_name").collect()}
        relations_dict = {
        row['relation_id']: row['relation_name'] 
        for row in edges_named.select("relation_id", "relation_name").collect()
    }
        return relations_dict,edges_named
def choose_id_LLM(relations_dict, included_triples, llm, sampling_params, style):
    prompt = f"""You are an AI evaluator.  Select only one triple in triples dictionary that are suitable to continue generating {style} based on {included_triples}. Output the keys as Qx_Px_Qx.
                
                relations:
                {relations_dict}
                
                output:"""
    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0])
    x = outputs[0].outputs[0].text.strip()
    print(x)
    x=re.findall('Q\d+\_P\d+\_Q\d+',x)
    return x[0]
    # ids = [x.strip() for x in x.split(',') if x.strip()]
    # return ids[0]
def choose_satrt_node(node_set, included_triples, llm, sampling_params, style):
    prompt = f"""You are an AI evaluator. Select only one start node to to continue generating {style} based on {included_triples}. Output the keys as Qx.
                
                nodes:
                {node_set}
                
                output:"""
    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0])
    x = outputs[0].outputs[0].text.strip()
    print(x)
    x=re.findall('Q\d+',x)
    return x[0]
    # ids = [x.strip() for x in x.split(',') if x.strip()]
    # return ids[0]
with open(entity_path, "rb") as f:
    rebel_entities = pickle.load(f)
with open(f"relations_A_{style}_new400.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["triples"])  # 只写一列，每行一个 relation list

    # 取前100个 src 节点
    start_nodes = set(list(rebel_entities)[201:400])
    # start_nodes =set([row['src'] for row in graph_simple_gf.edges.select('src').limit(3000).collect()])
    for start_node in start_nodes:
        entities_in_set={start_node}
        try:
            included_triples=[]
            included_triples_ids=[]
            for r in range(np.random.poisson(3)+1): # number satisfy Poisson
                current_src = choose_satrt_node(node_set=entities_in_set, included_triples=included_triples, llm=llm, sampling_params=sampling_params, style = style) # random choose next id 
                print(current_src)
                relations_dict,edges_named=filter_df(graph_simple_gf,current_src)
                for key_to_remove in included_triples_ids:
                    relations_dict.pop(key_to_remove, None)
                # next_triple_ids = random.choice(list(set(entities_in_set))) # random choose next id 
                # print(current_src)
                # relations_dict,edges_named=filter_df(graph_simple_gf,next_triple_ids)
                # if len(relations_dict)>20:
                #     all_keys = list(relations_dict.keys())
                #     random_keys = random.sample(all_keys, 20)
                #     relations_dict = {key: relations_dict[key] for key in random_keys}
                # relations_dict.pop(next_triple_ids,None)
                # print("iteration"+str(r))
                next_triple_ids = choose_id_LLM(relations_dict, included_triples, llm, sampling_params, style)
                # print("next_triple_ids"+next_triple_ids)
                # next_rel_id = re.findall("P\d+", next_rel_id)[0]
                # same_src_rel_def= edges_named.filter(F.lower(F.col("prop"))==next_rel_id.lower())
                # rows_list = same_src_rel_def.select("relation_name").collect()
                # objs=same_src_rel_def.select("dst").collect()
                # string_list = [row['relation_name'] for row in rows_list]
                # objs= [row['dst'] for row in objs]
                # included_triples.extend(string_list)
                # entities_in_set.extend(objs)
                # print(included_triples)
                # print(entities_in_set)
        
                if relations_dict is not None: 
                    print("iteration"+str(r))
                    next_triple_ids = choose_id_LLM(relations_dict, included_triples, llm, sampling_params, style)
                    print("next_triple_ids"+next_triple_ids)
                    # next_rel_id = re.findall("P\d+", next_rel_id)[0]
                    # same_src_rel_def= edges_named.filter(F.lower(F.col("prop"))==next_rel_id.lower())
                    # rows_list = same_src_rel_def.select("relation_name").collect()
                    # objs=same_src_rel_def.select("dst").collect()
                    # string_list = [row['relation_name'] for row in rows_list]
                    # objs= [row['dst'] for row in objs]
                    # included_triples.extend(string_list)
                    # entities_in_set.extend(objs)
                    # print(included_triples)
                    # print(entities_in_set)
                    entities_in_set.update(re.findall("Q\d+", next_triple_ids))
                    included_triples.append(relations_dict[next_triple_ids])
                    included_triples_ids.append(next_triple_ids)
                    print(entities_in_set)
                    print(included_triples)
                    print(included_triples_ids)
                else:
                    print("none")
                    included_triples.append("none")
                    break
            writer.writerow([included_triples])
        except Exception as e:
            writer.writerow([included_triples])
            print(e)