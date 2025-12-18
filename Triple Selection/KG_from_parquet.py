from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, when
import networkx as nx
import pandas as pd
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import networkx as nx
import pickle
from datetime import date
import numpy as np
import traceback
import types
import argparse
from datetime import datetime
import os
from pyspark.sql import functions as F
from graphframes import GraphFrame

def load_dictionaries(spark, rel_path,en_map,entity_path):
    with open(en_map, "rb") as f:
        english_dict = pickle.load(f)
        en_mapping_list = list(english_dict.items())
        mapping_df = spark.createDataFrame(en_mapping_list, ["id", "label"])
    with open(rel_path, "rb") as f:
        relations = pickle.load(f)
        relations_data = [(pid,) for pid in relations]
        # 2. 现在创建 DataFrame
        rel_df = spark.createDataFrame(relations_data, ["id"])
    with open(entity_path, "rb") as f:
        entities = pickle.load(f)
        entities = [(qid,) for qid in entities]
        entity_df=spark.createDataFrame(entities, ["id"])

    return rel_df, mapping_df, entity_df

def create_graphframe_from_spark(spark, all_rows_df, 
                                 nodes_mapping_df, 
                                 relations_filter_df, 
                                 nodes_filter_df, 
                                 simple=True):
    """
    从 Spark DataFrame (Parquet) 分布式地创建 GraphFrame。
    
    - all_rows_df: 包含所有三元组的原始 Spark DataFrame
    - nodes_mapping_df: (id, label) - 用于给节点添加标签
    - relations_filter_df: (id) - 用于过滤边 (只保留在 set 中的)
    - nodes_filter_df: (id) - 用于过滤节点 (只保留在 set 中的)
    """

    if simple:
        sub_col = "simple_sub_qid"
        obj_col = "simple_obj_qid"
        edge_col = "simple_edge_pid"
    else:
        # ... (你的 complex_... 列名)
        sub_col = "complex_sub_qid"
        obj_col = "complex_obj_qid"
        edge_col = "complex_edge_pid"

    # 1. 选择基础的 边 DataFrame
    print("Selecting base edges...")
    base_edges_df = all_rows_df.select(
        F.col(sub_col).alias("src"),
        F.col(obj_col).alias("dst"),
        F.col(edge_col).alias("relation_id")
    )
    print("base_edges_df.count()",base_edges_df.count())

    base_edges_df = base_edges_df.withColumn(
        "parsed_pid", 
        F.split(F.col("relation_id"), "_").getItem(1)
    )

    # 2. 过滤 Edges (边)
    # 这是一个三向 Join，Spark 会优化它。
    
    # 2a. 过滤：只保留 relation_id 在 relations_filter_df 中的边
    print("Filtering edges by relation_set...")
    filtered_edges = base_edges_df.join(
        F.broadcast(relations_filter_df),
        base_edges_df.parsed_pid == relations_filter_df.id,
        "inner"  # "inner" join = 只保留匹配项
    )
    print("filtered_edges",filtered_edges.count())
    # 2b. 过滤：只保留 src (源节点) 在 nodes_filter_df 中的边
    print("Filtering edges by nodes_set (source)...")
    filtered_edges = filtered_edges.join(
        F.broadcast(nodes_filter_df),
        filtered_edges.src == nodes_filter_df.id,
        "inner" # "inner" join = 只保留匹配项
    ).drop(nodes_filter_df.id)
    print("filtered_edges",filtered_edges.count())
    # 2c. 过滤：只保留 dst (目标节点) 在 nodes_filter_df 中的边
    print("Filtering edges by nodes_set (destination)...")
    filtered_edges = filtered_edges.join(
        F.broadcast(nodes_filter_df),
        filtered_edges.dst == nodes_filter_df.id,
        "inner" # "inner" join = 只保留匹配项
    )
    print("filtered_edges",filtered_edges.count())
    # 最终的边 DataFrame
    # edges_df = filtered_edges.select("src", "dst", "parsed_pid").distinct()
    edges_df = filtered_edges.select("src", "dst", "relation_id")
    # 3. 创建 Vertices (节点) DataFrame
    # 既然我们有完整的节点列表 (nodes_set)，我们直接使用它
    print("Creating Vertices DataFrame from nodes_set...")
    vertices_df = nodes_filter_df # 这已经有了 "id" 列
    
    # 4. 为节点添加标签 (Name)
    # 使用 "left" join，这样即使某个节点在 english_dict 中没有名字，节点本身也会被保留
    vertices_df = vertices_df.join(
        F.broadcast(nodes_mapping_df),
        "id",
        "left"
    ).select("id", F.col("label").alias("name")) # 重命名为 "name" 或你喜欢的
    print("filteredv-_edges",vertices_df.count())
    # 5. 创建 GraphFrame
    print("Creating GraphFrame object...")
    g = GraphFrame(vertices_df, edges_df)
    # g.cache()
    
    return g




        