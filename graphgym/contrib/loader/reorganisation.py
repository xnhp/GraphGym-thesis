import itertools
import os

import networkx as nx
from data.models import SBMLModel, SpeciesAliasId
from data.util import is_model_file, groupby, SpeciesClass
from deprecated.classic import deprecated
from graphgym.contrib.feature_augment.util import split_rxn_nodes, nx_get_interpretations, get_prediction_nodes
from more_itertools import powerset
from networkx import NetworkXError

import deepsnap.graph

dup_count = 0
new_node_count = 0


def flatten(l: list[list]):
    return list(itertools.chain(*l))


def set_labels_by_step(curr_g: nx.Graph, next_g: nx.Graph):
    # TODO doing this twice for both bipartite projection and simple graph?
    """
    Set node target labels of `curr_g` based on whether nodes were duplicated, producing `next_g`.
    # TODO description of criteria
    # TODO note that we are only considering splits along complete hyperedges, i.e. do not introduce
    #   duplicate on same reaction (but even then the approach would work I think)
    """
    # cannot compare models because the model may be the same, only the graph is constructed differently
    #   (might as well have implemented collapsing on the model)
    next_mdl = next_g.graph['model']
    next_g_simple, next_g_proj = nx_get_interpretations(next_g)
    next_g_use = next_g_simple
    curr_mdl = curr_g.graph['model']
    curr_g_simple, curr_g_proj = nx_get_interpretations(curr_g)
    curr_g_use = curr_g_simple

    # species that have new aliases
    # new_aliases_species = set([alias['species'] for alias in new_aliases])

    # group new aliases by species
    # assert curr_mdl.alias_groupby_attrib == next_mdl.alias_groupby_attr and curr_mdl.alias_groupby_attrib is not None
    # keys: species id; items: alias dicts representing that species
    # new_aliases_grouped = groupby(new_aliases, lambda x: x[curr_mdl.alias_groupby_attrib])

    # identify all duplications from curr_g to next_g
    # our approach is to consider newly introduced aliases in next_g and try to identify
    #   their "duplication parents" (if any), i.e. nodes that were split into this alias
    # assumes that node ids correspond to species alias ids. this is not always the case since
    #   wrapping a networkx graph in a deepsnap graph may trigger re-labeling of the nodes with sequential integers
    curr_g_directed = curr_g.graph['nx_multidigraph']
    next_g_directed = next_g.graph['nx_multidigraph']

    def maybe_get_parent(target_alias):

        def neighbs_safe(g, func, node_id):
            if node_id in g:
                return func(node_id)
            else:
                return []

        def successors_safe(g, node_id):
            return neighbs_safe(g, g.successors, node_id)

        def predecessors_safe(g, node_id):
            return neighbs_safe(g, g.predecessors, node_id)

        # check if has duplication parent, add to map
        # P_plus := N_t^(-) ( N_t+1^(+) ( v_i ) ) \ N_t+1^(-) ( N_t+1^(+) ( v_i ) )
        successors_in_next = successors_safe(next_g_directed, target_alias['id'])
        toNodest = set(flatten([
            predecessors_safe(curr_g_directed, alias)
            for alias in successors_in_next
        ]))
        P_plus = toNodest.difference(set(flatten([
            predecessors_safe(next_g_directed, alias)
            for alias in successors_in_next
        ])))
        # candidates must be of same class/type
        P_plus = set(filter(
            lambda alias: curr_g.nodes[alias]['class'] == target_alias['class'],
            P_plus
        ))

        predecessors_in_next = predecessors_safe(next_g_directed, target_alias['id'])
        fromNodest = set(flatten([
            successors_safe(curr_g_directed, alias)
            for alias in predecessors_in_next
        ]))
        P_minus = fromNodest.difference(set(flatten([
            successors_safe(next_g_directed, alias)
            for alias in predecessors_in_next
        ])))
        P_minus = set(filter(
            lambda alias: curr_g.nodes[alias]['class'] == target_alias['class'],
            P_minus
        ))

        if len(P_plus) == 0 and len(P_minus) == 0:
            return target_alias, None

        if len(toNodest) == 0 or len(fromNodest) == 0:
            P = set.union(P_plus, P_minus)  # take non-empty one
        else:
            # if node was not a source or sink, a valid duplication parent
            #   must not be source or sink, i.e. must appear in both sets
            P = set.intersection(P_plus, P_minus)

        # must represent same species
        P = set(filter(lambda a: curr_g.nodes[a]['species'] == target_alias['species'], P))

        # may be empty if no parent found
        # may be greater than 1 if edges were moved between duplicates
        # TODO what to do if more than 1 candidate?
        parent = P.pop() if len(P) == 1 else None
        return target_alias, parent

    # must not do this based on models here in case we are comparing with a collapsed graph
    #   (a collapsed graph is based on the same model, just the graph is constructed differently)
    # use proj as to not consider reactions # TODO express this more nicely
    new_node_ids = [node for node in next_g_proj if node not in curr_g_proj]
    # ↑ not including reactions
    # assume node id is alias id — fetch alias info based on node id
    new_aliases = [next_mdl.aliases[new_node_id] for new_node_id in new_node_ids]
    new_aliases = [alias for alias in new_aliases if not alias['species'] in [SpeciesClass.reaction.value, SpeciesClass.complex.value]]
    print(f"{curr_g_use.name} \t {len(new_aliases)} new alias nodes in next graph (not rxn or complex)")

    duplications: dict[SpeciesAliasId, list[SpeciesAliasId]] = groupby(
        [x for x in map(maybe_get_parent, new_aliases) if x[1] is not None],
        lambda t: t[1]  # group by id of duplication parent
    )
    print(f"{curr_g.name} \t {len(duplications.keys())} duplication parents identified")

    def has_dups(target_alias) -> bool:
        return is_dup_parent(target_alias)

    def is_dup_parent(candidate_parent_alias):
        # return candidate_parent_alias['id'] in duplications.keys()
        return candidate_parent_alias in duplications.keys()

    # set labels in curr_g based on whether a node has duplicates or not.
    prediction_targets, _ = get_prediction_nodes(curr_g)
    prediction_targets = set(prediction_targets)
    for _, node in curr_g.nodes.items():
        if node['id'] in prediction_targets:
            node['node_label'] = int(has_dups(node['id']))
        else:
            # note we exclude these later (see usages of `exclude_node_labels`)
            # but can already avoid expensive computation here.
            node['node_label'] = 0


def has_dups_subset(new_aliases_grouped, curr_g_use, next_g_use, target_alias):
    """
    Second approach
    :param new_aliases_grouped:
    :param curr_g_use:
    :param next_g_use:
    :param target_alias:
    :return:
    """
    # new aliases that represent the same species
    try:
        new_aliases_for_species = new_aliases_grouped[target_alias['species']]
    except KeyError:
        # there are no newly introduced aliases for the species that target_alias corresponds to
        #   this means it cannot have been duplicated in this step
        # return False
        pass
    # gather all new aliases of same species whose neighbourhood (in next_g) is a subset
    #   of neighbourhood of target_alias in curr_g
    old_neighbourhood = set(curr_g_use.neighbors(target_alias['id']))
    new_adjacent = [alias for alias in new_aliases_for_species
                    if set(next_g_use.neighbors(alias['id'])).issubset(old_neighbourhood)]
    if len(new_adjacent) == 0:
        return False
    else:
        # a motivation for this relaxation might be that there are possibly other nodes newly introduced
        # in the neighbourhood
        return True
    neighbours_of_new = list(itertools.chain(*[next_g_use.neighbors(new['id']) for new in new_adjacent]))
    try:
        neighbours_of_target_in_next = list(next_g_use.neighbors(target_alias['id']))
    except:
        print(f"could not find node {target_alias['id']} in next_g")
        neighbours_of_target_in_next = []
    # ensure that proper partition of incident edges / neighbourhood: no superfluous edges
    # ↓ neighbourhood in new graph
    new_neighbourhood = neighbours_of_new + list(neighbours_of_target_in_next)  # note: multiset
    # need to only count lengths because already established that subsets
    return len(new_neighbourhood) == len(old_neighbourhood)


def has_dups_powerset(next_g, curr_g):
    """
    first approach
    :param next_g:
    :param curr_g:
    :return:
    """

    def safe_neighbs(nxG, node):
        try:
            return list(nxG.neighbors(node))
        except NetworkXError:
            print(f"could not get neighbours of {node}")
            return []

    # an alias is marked as to-be-duplicated if in next_g there are >= 2 aliases, at least one of them new...
    # (>= condition already given by that we only look at *new* aliases (?`))
    # TODO this seems to take fairly long for e.g. PDMap
    #   we dont need to consider all powersets but for a given species only those that are new and also adjacent
    candidate_dup_results = filter(
        lambda s: len(s) > 0,
        powerset(new_aliases_for_species))
    # ... whose adjacencies exactly make up the adjacency of the alias
    # (need to also consider alias that was already there)
    candidate_dup_results = [
        # append target_alias to tuples
        subset + (target_alias,) for subset in candidate_dup_results
    ]

    def are_dups(candidate_subset):
        # are the aliases in candidate_subset dups of `alias`?
        # i.e. union of neighbours in next_g is neighbourhood of alias in curr_g
        # TODO speed up by using boolean vector operations?
        neighbs_union = set.union(*[
            set(
                # consider only ids because nodes might differ in other attributes (or hash value) between graphs
                [nb for nb in safe_neighbs(next_g, a['id'])]  # already returns only ids i think
            ) for a in candidate_subset
        ])
        neighbs_target = set([nb for nb in curr_g.neighbors(target_alias['id'])])
        return neighbs_union == neighbs_target  # compares elements

    return any(map(are_dups, candidate_dup_results))


def load_reorganisation_steps(dataset, loader) -> list[deepsnap.graph.Graph]:
    """
    :param dataset:
    :param loader: The proper loader for the graph interpretation as given by config yaml
    :return:
    """
    collection_path, model_class = dataset
    return [loader(entry.path, model_class, entry.name)
            for entry in os.scandir(collection_path)
            if is_model_file(entry)]


def set_labels_all_none(nxG):
    for node in nxG.nodes:
        node['node_label'] = 0


@deprecated(reason="this approach will no longer be considered, compare to collapsed graph instead")
def set_labels_by_duplicate_alias(nxG):
    """
    Set classification target labels based on whether for a node representing a speciesAlias there exist
    other speciesAliases corresponding to the same species.
    This is the same as considering a reorganisation step from a "collapsed" graph in which all duplicate speciesAliases
    are collapsed into a single representation.
    :param nxG:
    :return:
    """
    rxn_node_ids, other_ids = split_rxn_nodes(nxG)
    model: SBMLModel
    model = nxG['model']
    # assume all non_reaction nodes are species
    for species_alias_id, _ in other_ids:
        nxG.nodes[species_alias_id]['node_label'] = int(
            species_alias_id in model.species_aliases_with_duplicates
        )
    for rxn_id, _ in rxn_node_ids:
        nxG.nodes[rxn_id]['node_label'] = 0
