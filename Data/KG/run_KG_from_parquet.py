import KG_from_parquet as KGF
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
from datetime import datetime
from pyspark.storagelevel import StorageLevel
import torch

rel_path = "rebel_relations.pkl"
en_map = "mapping.pkl"
entity_path = "rebel_entities.pkl"


conf = (
    pyspark.SparkConf()
    .setAppName("WikipediaProcessing")
    .setMaster("local[10]")  
    .setAll(
        [
            ("spark.driver.memory", "250g"),
            ("spark.driver.maxResultSize", "32G"),
            ("spark.memory.fraction", "0.75"),
            ("spark.sql.shuffle.partitions", "8000"), # Try 2000 or 4000
            # --- ADD MEMORY OVERHEAD ---
            ("spark.driver.memoryOverhead", "8g") # Add some overhead memory
        ]
    )
)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
rel_df, mapping_df, entity_df = KGF.load_dictionaries(spark,rel_path,en_map,entity_path)
all_rows = spark.read.parquet("./test_output/graph_data_all.parquet")
# SKEWED_KEYS_TO_FILTER = [
#     'Q5', 'Q30', 'Q1860', 'Q145', 'Q2736', 'Q6655'
# ]
# all_rows = all_rows.filter(
#     ~col("simple_sub_qid").isin(SKEWED_KEYS_TO_FILTER) & \
#     ~col("simple_obj_qid").isin(SKEWED_KEYS_TO_FILTER)
# )
graph_simple_gf = KGF.create_graphframe_from_spark(
    spark, 
    all_rows, 
    mapping_df, 
    rel_df, 
    entity_df, 
    simple=True
)

num_nodes = graph_simple_gf.vertices.count()
num_edges = graph_simple_gf.edges.count()

print("Number of nodes (filtered):")
print(num_nodes)
print("Number of edges (filtered):")
print(num_edges)


def LLM_select_ids(src, relation_set, tokenizer, model):
    messages = [
        {"role": "system", "content": f"You are an AI evaluator. Select only one relation that is suitable to generate news for {src}."},
        {"role": "user", "content": f"Relations:\n{relation_set}\n\nSelect only one relations. Only output the realtion's id. Do NOT add extra information."}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False).strip()
    
    #Extract text after the last [/INST] or system prompt
    if "[/inst]" in generated_text.lower():
        response = generated_text.lower().split("[/inst]")[-1].strip()
    else:
        response = generated_text.strip()
    ids = re.findall(r'\bp\d+\b', response)
    return ids


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

# 添加列并映射
edges_named = (
    reloaded_graph_gf.edges
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



# output_dir = "./test_output/simple_graph_df"
# #--
# shuffle_partitions = int(spark.conf.get("spark.sql.shuffle.partitions"))
# # 1. 保存节点 DataFrame
# print(f"Saving vertices to {output_dir}/vertices ...")
# # try:
# #     vertices_to_save = graph_simple_gf.vertices.repartition(shuffle_partitions, "id")
# #     vertices_to_save.write.mode("overwrite").parquet(f"{output_dir}/vertices")
# # except Exception as e:
# #         print(f"Error saving vertices: {e}")
# # graph_simple_gf.vertices.write.mode("overwrite").parquet(f"{output_dir}/vertices")
# # 2. 保存边 DataFrame
# print(f"Saving edges to {output_dir}/edges ...")
# SALTING_FACTOR = 500
# # graph_simple_gf.edges.write.mode("overwrite").parquet(f"{output_dir}/edges")
# try:
#     # edges_to_save = graph_simple_gf.edges.repartition(shuffle_partitions, "src", "dst")
#     # edges_to_save.write.mode("overwrite").parquet(f"{output_dir}/edges")
#     edges_with_salt = graph_simple_gf.edges.withColumn(
#             "_salt", 
#             (floor(rand() * SALTING_FACTOR)).cast("int") 
#         )

#     print(f"Repartitioning edges to {shuffle_partitions} partitions using SALT, SRC, DST before saving...")
#     # Repartition based on the salt AND the original keys
#     edges_to_save = edges_with_salt.repartition(shuffle_partitions, "_salt", "src", "dst") 
    
#     print(f"Saving edges to {output_dir}/edges ...")
#     # Write the salted and repartitioned data, dropping the salt column during write
#     edges_to_save.drop("_salt").write.mode("overwrite").parquet(f"{output_dir}/edges")
#     print("Edges saved.")
# except Exception as e:
#         print(f"Error saving vertices: {e}")

print("Stopping Spark session...")
spark.stop()
print("Spark session stopped.")