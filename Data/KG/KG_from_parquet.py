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
    create GraphFrame from Spark DataFrame (Parquet)
    
    - all_rows_df: original Spark DataFrame
    - nodes_mapping_df: (id, label) 
    - relations_filter_df: only keep relations from rebel
    - nodes_filter_df: only keep nodes from rebel
    """

    if simple:
        sub_col = "simple_sub_qid"
        obj_col = "simple_obj_qid"
        edge_col = "simple_edge_pid"
    else:
        sub_col = "complex_sub_qid"
        obj_col = "complex_obj_qid"
        edge_col = "complex_edge_pid"


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

    print("Filtering edges by relation_set...")
    filtered_edges = base_edges_df.join(
        F.broadcast(relations_filter_df),
        base_edges_df.parsed_pid == relations_filter_df.id,
        "inner" 
    )
    print("filtered_edges",filtered_edges.count())

    print("Filtering edges by nodes_set (source)...")
    filtered_edges = filtered_edges.join(
        F.broadcast(nodes_filter_df),
        filtered_edges.src == nodes_filter_df.id,
        "inner" 
    ).drop(nodes_filter_df.id)
    print("filtered_edges",filtered_edges.count())
    # 2c. only keep dst in nodes_filter_df 
    print("Filtering edges by nodes_set (destination)...")
    filtered_edges = filtered_edges.join(
        F.broadcast(nodes_filter_df),
        filtered_edges.dst == nodes_filter_df.id,
        "inner" 
    )
    print("filtered_edges",filtered_edges.count())
    # edges_df = filtered_edges.select("src", "dst", "parsed_pid").distinct()
    edges_df = filtered_edges.select("src", "dst", "relation_id")
    print("Creating Vertices DataFrame from nodes_set...")
    vertices_df = nodes_filter_df # 这已经有了 "id" 列
    
    vertices_df = vertices_df.join(
        F.broadcast(nodes_mapping_df),
        "id",
        "left"
    ).select("id", F.col("label").alias("name")) 
    print("filteredv-_edges",vertices_df.count())

    print("Creating GraphFrame object...")
    g = GraphFrame(vertices_df, edges_df)
    # g.cache()
    
    return g




        