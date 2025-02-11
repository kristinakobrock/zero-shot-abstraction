from collections import Counter
from utils.analysis_from_interaction import *


def objects_to_concepts(sender_input, n_values):
    """reconstruct concepts from objects in interaction"""
    n_targets = int(sender_input.shape[1]/2)
    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects, all_targets=True)
    concepts = list(zip(objects, fixed))
    return concepts


def retrieve_messages(interaction):
    """retrieve messages from interaction"""
    messages = interaction.message.argmax(dim=-1)
    messages = [msg.tolist() for msg in messages]
    return messages


def count_symbols(messages):
    """counts symbols in messages"""
    all_symbols = [symbol for message in messages for symbol in message]
    symbol_counts = Counter(all_symbols)
    return symbol_counts


def get_unique_message_set(messages):
    """returns unique messages as a set ready for set operations"""
    return set(tuple(message) for message in messages)


def get_unique_concept_set(concepts):
    """returns unique concepts"""
    concept_tuples = []
    for objects, fixed in concepts:
        tuple_objects = []
        for obj in objects:
            tuple_objects.append(tuple(obj))
        tuple_objects = tuple(tuple_objects)
        tuple_concept = (tuple_objects, tuple(fixed))
        concept_tuples.append(tuple_concept)
    tuple(concept_tuples)
    unique_concepts = set(concept_tuples)
    return unique_concepts


def look_up_values(index_vector, value_vector, dictionary):
    """
    Look up values in a dictionary for index-attribute pairs from two vectors.

    Args:
        index_vector (list): A list of indices.
        value_vector (list): A list of values corresponding to the indices.
        dictionary (dict): A dictionary with index-attribute pairs as keys.

    Returns:
        list: A list of looked-up values.
    """
    # Initialize an empty list to store the looked-up values
    looked_up_values = []

    # Iterate over the index and value pairs
    for index, value in zip(index_vector, value_vector):
        # Construct the key by concatenating the index and value as strings
        key = f"{index}{value}"

        # Look up the value in the dictionary and append it to the list
        looked_up_values.append(dictionary.get(key))

    return looked_up_values
