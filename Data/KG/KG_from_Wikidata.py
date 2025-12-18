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

DISAMBIGUATION = "Q4167410"
LIST = "Q13406463"
INTERNAL_ITEM = "Q17442446"
CATEGORY = "Q4167836"
literal_dict = {}
nodes_ids = []
edges_ids = []
nodes = []
edges = []


def create_graph(nodes, edges):
    # nodes contain qid and name
    # edges contain sub_qid, obj_qid, pid and rel (name)
    graph = nx.Graph()
    nodes = [(elem["qid"], elem) for elem in nodes]
    edges = [(elem["sub_qid"], elem["obj_qid"], {"pid": elem["pid"], "name": elem["rel"]}) for elem in edges]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def extract_unit_dict(unit_path):
    # process the JSON file
    unit_dict = {}
    with open(unit_path, "r") as f:
        json_dict = json.load(f)
    for key, d in json_dict.items():
        unit_dict[key] = d["label"]
    return unit_dict


def assign_literal_id(literal):
    return literal
    # if literal is None:
    #     return None
    # global literal_dict
    # if literal in literal_dict:
    #     return literal_dict[literal]
    # lid = "L" + str(id(literal))
    # literal_dict[literal] = lid
    # return lid


def assign_complex_node_id(sub_qid, rel, obj_qid):
    return sub_qid + "_" + rel + "_" + obj_qid


def dd2dms(deg):
    d = int(deg)
    md = np.abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]


def transform_time(time_dict):  # TODO check how they handle timezones
    # extract the "(-+)yyyy-mm-dd as hours, minutes and seconds aren't used
    value = ""
    time_string = time_dict["time"].split("T")[0]
    bce = True if time_string[0] == "-" else False
    try:
        year, month, day = time_string.split("-")
    except Exception as e:
        return None
    # remove (-+) and zeros in front of the year
    year = year[1:].lstrip("0")
    precision = time_dict["precision"]
    if precision == 11:
        # day
        date_obj = date(int(year), int(month), int(day))
        value = date_obj.strftime("%d %B %Y")
    elif precision == 10:
        # month
        date_obj = date(int(year), int(month), 1)
        value = date_obj.strftime("%B %Y")
    elif precision == 9:
        # year
        date_obj = date(int(year), 1, 1)  # TODO ValueError: year 19356 is out of range
        value = date_obj.strftime("%Y")
    elif precision == 8:
        # decade
        value = str(year) + "s"
    elif precision == 7:
        century = int(year) / 100 + 1
        value = str(century) + ". century"
    elif precision == 6:
        mill = int(year) / 1000
        value = str(mill) + ". millennium"
    else:
        return None
    if bce:
        value += " BCE"
    return value


def transform_quantity(quantity_dict):
    value = quantity_dict["amount"]
    if value.startswith("+"):
        value = value[1:]
    unit = quantity_dict["unit"]
    if unit.startswith("http"):
        unit = unit.split("/")[-1]
        try:
            unit = unit
        except Exception as e:
            unit = None
    else:
        unit = None
    if unit is not None:
        return value + " " + unit
    else:
        return value


def transform_coordinate(coordinate_dict):
    latitude = coordinate_dict["latitude"]
    longitude = coordinate_dict["longitude"]
    lat_dir = "N" if latitude > 0 else "S"
    lon_dir = "E" if longitude > 0 else "W"
    lat_d, lat_m, lat_s = dd2dms(np.abs(latitude))
    lon_d, lon_m, lon_s = dd2dms(np.abs(longitude))  # TODO check if it needs to be absolute val
    value = (
        str(lat_d)
        + "°"
        + str(lat_m)
        + "'"
        + str(lat_s)
        + '"'
        + lat_dir
        + ", "
        + str(lon_d)
        + "°"
        + str(lon_m)
        + "'"
        + str(lon_s)
        + '"'
        + lon_dir
    )
    return value


def extract_object(claim):
    try:
        if not ("datavalue" in claim):
            return None, None
        if claim["datatype"] == "wikibase-item":
            obj_qid = claim["datavalue"]["value"]["id"]  # for wikibase items
            if obj_qid in english_dict:
                val = english_dict[obj_qid]
            else:
                return None, None  # both of the entities have to be from English WP
        # object is a literal
        elif claim["datatype"] == "string":
            val = claim["datavalue"]["value"]
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "time":
            # convert time to the format
            val = transform_time(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "globe-coordinate":
            # convert the coordinate
            val = transform_coordinate(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "quantity":
            # convert the quantity
            val = transform_quantity(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "external-id":
            # convert the external id
            val = claim["datavalue"]["value"]
            obj_qid = assign_literal_id(val)
        else:
            # filter out other types of literals
            return None, None
        return obj_qid, val
    except Exception as e:
        traceback.print_exc()
        return None, None


def process_qualifiers(
    claim, sub_qid, sub_name, relation_pid, relation_name, obj_qid, obj_name, rows, use_qualifiers=False
):
    if not use_qualifiers:
        rows.append(
            Row(
                simple_sub={"qid": sub_qid, "name": sub_name},
                simple_sub_qid=sub_qid,
                simple_obj={"qid": obj_qid, "name": obj_name},
                simple_obj_qid=obj_qid,
                simple_edge={"pid": relation_pid, "sub_qid": sub_qid, "rel": relation_name, "obj_qid": obj_qid},
                simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
            )
        )
        return rows
    complex_node_id = assign_complex_node_id(sub_qid, relation_pid, obj_qid)
    if "qualifiers" in claim:
        for qualifier, qualifier_list in claim["qualifiers"].items():
            if qualifier in relations:
                qualifier_name = english_dict[qualifier]
                for qual_elem in qualifier_list:
                    complex_obj_qid, complex_obj_name = extract_object(qual_elem)  # TODO
                    if complex_obj_qid is not None and complex_obj_name is not None:
                        complex_edge = {
                            "pid": qualifier,
                            "sub_qid": complex_node_id,
                            "rel": qualifier_name,
                            "obj_qid": complex_obj_qid,
                        }
                        complex_sub = {
                            "qid": complex_node_id,
                            "complex_sub_qid": sub_qid,
                            "complex_sub_name": sub_name,
                            "complex_pid": relation_pid,
                            "complex_rel": relation_name,
                            "complex_obj_qid": obj_qid,
                            "complex_obj_name": obj_name,
                        }
                        complex_obj = {"qid": complex_obj_qid, "name": complex_obj_name}
                        complex_rel_qid = complex_node_id + "_" + qualifier + "_" + complex_obj_qid
                    else:
                        complex_sub, complex_obj, complex_edge = None, None, None
                        complex_rel_qid = None
                    rows.append(
                        Row(
                            simple_sub={"qid": sub_qid, "name": sub_name},
                            simple_sub_qid=sub_qid,
                            simple_obj={"qid": obj_qid, "name": obj_name},
                            simple_obj_qid=obj_qid,
                            simple_edge={
                                "pid": relation_pid,
                                "sub_qid": sub_qid,
                                "rel": relation_name,
                                "obj_qid": obj_qid,
                            },
                            simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
                            complex_sub=complex_sub,
                            complex_sub_qid=complex_node_id,
                            complex_edge=complex_edge,
                            complex_edge_pid=complex_rel_qid,
                            complex_obj=complex_obj,
                            complex_obj_qid=complex_obj_qid,
                        )
                    )
            else:
                rows.append(
                    Row(
                        simple_sub={"qid": sub_qid, "name": sub_name},
                        simple_sub_qid=sub_qid,
                        simple_obj={"qid": obj_qid, "name": obj_name},
                        simple_obj_qid=obj_qid,
                        simple_edge={"pid": relation_pid, "sub_qid": sub_qid, "rel": relation_name, "obj_qid": obj_qid},
                        simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
                        complex_sub=None,
                        complex_sub_qid=None,
                        complex_edge=None,
                        complex_edge_pid=None,
                        complex_obj=None,
                        complex_obj_qid=None,
                    )
                )

    return rows



def process_claim(claim, subject_qid, subject_name, prop, rows, use_qualifiers):
    rel_name = english_dict[prop]
    obj_qid, obj_name = extract_object(claim["mainsnak"])
    if obj_qid is not None and obj_name is not None:
        # checking the qualifiers
        rows = process_qualifiers(
            claim, subject_qid, subject_name, prop, rel_name, obj_qid, obj_name, rows, use_qualifiers
        )
    return rows

def find_preferred(claims):
    normal_claims = []
    for claim in claims:
        if claim["rank"] == "preferred":
            return [claim]
        if claim["rank"] == "normal":
            normal_claims.append(claim)
    return normal_claims
# --- Add this function for cleaner debugging ---
def debug_print(stage, data):
    """Helper to print debugging info."""
    print(f"[DEBUG] Stage: {stage} | Data: {str(data)[:200]}")


def get_entity_info(line):
    # Initialize rows at the start
    rows = []
    
    # NOTE: You had two 'except Exception as e:' at the same level, which is a syntax error.
    # I have combined them into a single block.
    try:
        debug_print("START", line.strip())

        if DISAMBIGUATION in line or LIST in line or INTERNAL_ITEM in line or CATEGORY in line:
            debug_print("FAIL - Broad Filter", "")
            return []

        # Safely prepare the line for JSON parsing
        clean_line = line.strip()
        if clean_line.endswith(','):
            clean_line = clean_line[:-1]
        
        if not clean_line:
            return []

        row = json.loads(clean_line)
        debug_print("JSON Loaded", row.get("id"))

        if not ("type" in row and row["type"] == "item"):
            debug_print("FAIL - Not type 'item'", row.get("type"))
            return []

        subject_qid = row["id"]
        debug_print("Found item", subject_qid)
        
        if subject_qid not in english_dict:
            debug_print("FAIL - QID not in english_dict", subject_qid)
            return []
        
        subject_name = english_dict[subject_qid]
        debug_print("Found name in english_dict", subject_name)

        # Flag to see if we ever process a claim
        processed_at_least_one_claim = False

        for prop, claims in row.get("claims", {}).items():
            print(prop)
            if prop not in relations:
            #     # This is normal, so we don't print here to avoid spam
                 continue
            
            debug_print(f"Found matching property", prop)
            
            claims_to_process = find_preferred(claims)
            for claim in claims_to_process:
                # Assuming process_claim modifies and returns the rows list
                rows = process_claim(claim, subject_qid, subject_name, prop, rows, False)
                processed_at_least_one_claim = True

        if not processed_at_least_one_claim:
             debug_print("WARN - No valid properties found in relation_dict for this item", subject_qid)

        debug_print("FINAL ROWS", rows)
        return rows

    except json.JSONDecodeError as e:
        # Catch JSON errors specifically
        debug_print("FAIL - JSON Decode Error", f"Error: {e}, Line: {line.strip()}")
        return []
    except Exception as e:
        # Catch all other errors
        print("An unexpected error occurred:")
        traceback.print_exc()
        return []
def helper_func_nodes(x):
    if x is not None:
        if x["qid"] in nodes_ids:
            nodes.append(x)
            nodes_ids.remove(x["qid"])


def helper_func_edges(x):
    if x is not None:
        pid = x["sub_qid"] + "_" + x["pid"] + "_" + x["obj_qid"]
        if pid in edges_ids:
            edges.append(x)
            edges_ids.remove(pid)
def extract_graph_from_parquet(df, simple=True):
    if simple:
        sub_col = "simple_sub_qid"
        obj_col = "simple_obj_qid"
        edge_col = "simple_edge_pid"
    else:
        sub_col = "complex_sub_qid"
        obj_col = "complex_obj_qid"
        edge_col = "complex_edge_pid"
    global nodes_ids
    global nodes
    nodes_ids = list(pd.unique(df[[sub_col, obj_col]].values.ravel("K")))
    global edges_ids
    global edges
    edges_ids = list(pd.unique(df[edge_col]))
    df[sub_col[:-4]].apply(helper_func_nodes)
    df[obj_col[:-4]].apply(helper_func_nodes)
    df[edge_col[:-4]].apply(helper_func_edges)
    del df
    graph = create_graph(nodes, edges)
    return graph


def print_literals_stats(graph_simple):
    print("number of literals and entities")
    literals = 0
    entities = 0
    for u, v in graph_simple.nodes(data=True):
        if u.startswith("L"):
            literals += 1
        elif u.startswith("Q"):
            entities += 1
    print(literals)
    print(entities)
def load_dictionaries(args):
    with open(args.rel_path, "rb") as f:
        relations = pickle.load(f)

    with open(args.en_map, "rb") as f:
        english_dict = pickle.load(f)
    with open(args.unit_path, "rb") as f:
        entities = pickle.load(f)

    return relations, english_dict, entities
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def extract_graph_from_parquet_in_chunks(all_rows: pd.DataFrame, chunk_size=100000, output_dir="./test_output/graph_chunks"):

    os.makedirs(output_dir, exist_ok=True)

    num_rows = len(all_rows)
    num_chunks = (num_rows + chunk_size - 1) // chunk_size
    chunk_files = []

    for i in range(num_chunks):
        chunk_file = os.path.join(output_dir, f"graph_chunk_{i}.pkl")
        chunk_files.append(chunk_file)

        if os.path.exists(chunk_file):
            log(f"⚡ Chunk {i} already exists, skipping...")
            continue

        log(f"🔧 Processing chunk {i} / {num_chunks}")
        chunk_df = all_rows.iloc[i*chunk_size:(i+1)*chunk_size]

        G_chunk = extract_graph_from_parquet(chunk_df)

        with open(chunk_file, "wb") as f:
            pickle.dump(G_chunk, f)
        log(f"✅ Saved chunk {i} to {chunk_file}")


    log("🧱 Merging all graph chunks...")
    full_graph = nx.Graph()
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            G_chunk = pickle.load(f)
        full_graph = nx.compose(full_graph, G_chunk)
    log(f"🎉 Full graph created: {len(full_graph.nodes())} nodes, {len(full_graph.edges())} edges")

    return full_graph


if __name__ == "__main__":
    
    mock_args = types.SimpleNamespace()
    mock_args.rel_path = "rebel_relations.pkl"
    mock_args.en_map = "mapping.pkl"
    mock_args.unit_path = "rebel_entities.pkl"
    
    # IMPORTANT: Use a small sample of the wikidata dump for testing!
    # See instructions below on how to create one.
    # mock_args.dump_path = "latest-all.json.gz"
    mock_args.dump_path= "wikidata-sample-1000.json.gz"
    
    # mock_args.graph_path = "./test_output/final_graph_all.pkl"
    mock_args.graph_path = "./test_output/example_graph.pkl"
    # mock_args.graph_df = "./test_output/graph_data_all.parquet"
    mock_args.graph_df = "./test_output/example_data.parquet"
    os.makedirs("./test_output", exist_ok=True)
    
    relations, english_dict, entities = load_dictionaries(mock_args)
    log("🚀 Starting Spark session...")
    conf = (
        pyspark.SparkConf()
        .setAppName("WikipediaProcessing")
        .setMaster("local[10]")  
        .setAll(
            [
                ("spark.driver.memory", "230g"),
                ("spark.driver.maxResultSize", "32G"),
            ]
        )
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    #     # create the context
    # sc = spark.sparkContext
    # wikidata_all = sc.textFile(mock_args.dump_path)
    # row = wikidata_all.flatMap(get_entity_info)
    # all_rows = spark.createDataFrame(row)
    #  # saving and making networkx
    # all_rows.write.mode("overwrite").parquet(mock_args.graph_df)
    all_rows = spark.read.parquet("./test_output/example_data.parquet")
    # all_rows = all_rows.toPandas()
    log("to pandas done")
    graph_simple = extract_graph_from_parquet(all_rows)
    # graph_simple = extract_graph_from_parquet_in_chunks(all_rows)
    
    print("number of edges and nodes")
    print(len(graph_simple.edges()))
    print(len(graph_simple.nodes()))
    
    with open(mock_args.graph_path, "wb") as f:
        pickle.dump(graph_simple, f)
        