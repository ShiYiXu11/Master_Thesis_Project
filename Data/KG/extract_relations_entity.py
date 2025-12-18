import pickle
import ast
from datasets import load_dataset

def get_relations_entities_rebel_from_hf(dataset,entity_file, relation_file, mapping_file):
    rebel_entities = set()
    rebel_relations = set()
    id_name_map = {}
    
    for obj in dataset:
        for triplet in obj["triplets"]:
            subject = ast.literal_eval(triplet["subject"])
            predicate = ast.literal_eval(triplet["predicate"])
            objects= ast.literal_eval(triplet["object"])
            
            subject_id = subject.get("uri", "")
            subject_name = subject.get("surfaceform", "")
            predicate_id = predicate.get("uri", "")
            predicate_name = predicate.get("surfaceform", "")
            object_id = objects.get("uri", "")
            object_name = objects.get("surfaceform", "")
            
            if subject_id.startswith("Q"):
                rebel_entities.add(subject_id)
                if subject_name: 
                    id_name_map[subject_id] = subject_name
            if object_id.startswith("Q"):
                rebel_entities.add(object_id)
                if object_name:
                    id_name_map[object_id]=object_name
            if predicate_id.startswith("P"):
                rebel_relations.add(predicate_id)
                if predicate_name:
                    id_name_map[predicate_id]=predicate_name
    

    if entity_file:
        with open(entity_file, "wb") as f:
            pickle.dump(rebel_entities, f)
    if relation_file:
        with open(relation_file, "wb") as f:
            pickle.dump(rebel_relations, f)
    if mapping_file:
        with open(mapping_file, "wb") as f:
            pickle.dump(id_name_map, f)

dataset = load_dataset("martinjosifoski/SynthIE", "rebel", split="train")

get_relations_entities_rebel_from_hf(
    dataset, 
    entity_file="rebel_entities.pkl", 
    relation_file="rebel_relations.pkl",
    mapping_file="mapping.pkl"
)

