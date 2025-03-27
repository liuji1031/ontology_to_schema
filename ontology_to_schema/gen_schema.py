"""
This module provides functionality to convert OWL ontologies to LinkML schemas.

Classes:
    OwlImportEngine: An ImportEngine that takes schema-style OWL and converts
    it to a LinkML schema.

Functions:
    fix_class_definition_sequence: Fix the sequence of class definitions in a
    LinkML schema YAML file to follow the inheritance hierarchy.

The OwlImportEngine class includes methods to:
    - Initialize the engine with configuration and logging.
    - Reset internal state.
    - Read OWL files in functional syntax.
    - Extract prefixes from ontology documents or TTL files.
    - Process various OWL axioms and declarations.
    - Handle subclass relationships and slot usage.
    - Remove redundant and irrelevant slots.
    - Query agents for slot relevance and assignment.
    - Convert OWL ontologies to LinkML schemas.
    - Rename forbidden class names.
    - Add slots to classes based on domain and range.

The fix_class_definition_sequence function ensures that class definitions in a
LinkML schema YAML file follow the inheritance hierarchy by reordering them.
"""

from __future__ import annotations

import logging
import os
import pathlib
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import partial
from typing import Any, Dict, List, Set, Tuple, Union

import linkml_model as lm
import yaml
from funowl import (
    IRI,
    AnnotationAssertion,
    AnnotationPropertyDomain,
    AnnotationPropertyRange,
    AnonymousIndividual,
    Axiom,
    Class,
    ClassExpression,
    DataAllValuesFrom,
    DataExactCardinality,
    DataHasValue,
    DataMaxCardinality,
    DataMinCardinality,
    DataOneOf,
    DataProperty,
    DataPropertyDomain,
    DataPropertyRange,
    DataSomeValuesFrom,
    Datatype,
    DatatypeRestriction,
    DataUnionOf,
    Declaration,
    Literal,
    ObjectAllValuesFrom,
    ObjectExactCardinality,
    ObjectIntersectionOf,
    ObjectMaxCardinality,
    ObjectMinCardinality,
    ObjectProperty,
    ObjectPropertyDomain,
    ObjectPropertyExpression,
    ObjectPropertyRange,
    ObjectSomeValuesFrom,
    ObjectUnionOf,
    StringLiteralWithLanguage,
    SubAnnotationPropertyOf,
    SubClassOf,
    SubDataPropertyOf,
    SubObjectPropertyOf,
    TypedLiteral,
)
from funowl.converters.functional_converter import to_python
from funowl.ontology_document import Ontology, OntologyDocument
from linkml_runtime.linkml_model import SchemaDefinition
from rdflib import Graph
from schema_automator.importers.import_engine import ImportEngine
from schema_automator.utils.schemautils import write_schema

from ontology_to_schema.agent_assign_slot import AgentAssignSlot
from ontology_to_schema.agent_relevant_slot import AgentRelevantSlot

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


TOOL_NAME = "ontology_to_schema"
LINKML_TYPES = [
    lm.String,
    lm.String,
    lm.Integer,
    lm.Boolean,
    lm.Float,
    lm.Double,
    lm.Decimal,
    lm.Time,
    lm.Date,
    lm.Datetime,
    lm.Uriorcurie,
    lm.Uri,
    lm.Ncname,
    lm.Objectidentifier,
    lm.Nodeidentifier,
]

TYPES_MAP = {t.type_class_curie: t.type_name for t in LINKML_TYPES}


IGNORE_PREFIXES = [
    "sro",
    "xml",
    "xsd",
    "dcam",
    "admin",
    "swrl",
    "swrlb",
    "owl2xml",
    "rss",
    "content",
]

PREFIX_MAP = {
    "owl": "http://www.w3.org/2002/07/owl",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema",
    "dc": "http://purl.org/dc/elements/1.1",
    "deo": "http://purl.org/spar/deo",
    "orb": "http://purl.org/orb/1.0",
    "sro": "http://salt.semanticauthoring.org/ontologies/sro",
    "sao": "http://salt.semanticauthoring.org/ontologies/sao",
    "dcterms": "http://purl.org/dc/terms",
    "po": "http://purl.org/spar/po",
    "pattern": "http://www.essepuntato.it/2008/12/pattern",
    "doco": "http://purl.org/spar/doco",
    "foaf": "http://xmlns.com/foaf/0.1",
    "vs": "http://www.w3.org/2003/06/sw-vocab-status/ns",
    "doap": "http://usefulinc.com/ns/doap",
    "grddl": "http://www.w3.org/2003/g/data-view",
    "skos": "http://www.w3.org/2004/02/skos/core",
    "wot": "http://xmlns.com/wot/0.1",
    "vann": "http://purl.org/vocab/vann",
    "bio": "http://purl.org/vocab/bio/0.1",
    "cc": "http://web.resource.org/cc",
    "ov": "http://open.vocab.org/terms",
    "label": "http://purl.org/net/vocab/2004/03/label",
    "sdo": "http://salt.semanticauthoring.org/ontologies/sdo",
}

INV_PREFIX_MAP = {v: k for k, v in PREFIX_MAP.items()}

URI_FILE_MAP = {v: k + ".ofn" for k, v in PREFIX_MAP.items()}

EQUIVALENT_PREFIXES: dict[str, str] = {}
EQUIVALENT_URIS: dict[str, str] = {}
EQUIVALENT_URIS[
    "http://www.w3.org/2003/06/sw-vocab-status"
] = "http://www.w3.org/2003/06/sw-vocab-status/ns"


def read_ofn_file(file: str) -> Tuple[OntologyDocument, Ontology]:
    """Read an OWL file (*.ofn) and return the ontology document."""
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    doc = to_python(file, print_progress=False)
    ontology: Ontology = doc.ontology
    # ontology: Ontology
    if len(ontology.axioms) == 0:
        raise Exception(
            f"Empty ontology in {file} "
            + "(note: ontologies must be in functional syntax)"
        )
    return doc, ontology


def extract_prefixes(
    doc: OntologyDocument, name: str | None, rm_trailing_char=False
):
    """Extract prefixes from an ontology document."""
    prefix_dict = {}
    for p in doc.prefixDeclarations.as_prefixes():
        prefix_name = (
            p.prefixName if len(p.prefixName) > 0 or name is None else name
        )
        if not rm_trailing_char:
            prefix_dict[prefix_name] = p.fullIRI
        else:
            # remove trailing "#" or "/"
            s = str(p.fullIRI)
            while s[-1] in ["#", "/"]:
                s = s[:-1]
            prefix_dict[prefix_name] = s

    # also add the imports as prefix
    inv_prefix_dict = {v: k for k, v in prefix_dict.items()}
    return prefix_dict, inv_prefix_dict


def find_best_match(strings: List[str], target: str) -> str:
    """
    Find the string in the list that has the highest match with the target str.

    Args:
        strings (List[str]): List of strings to search.
        target (str): The target string to match against.

    Returns:
        str: The string with the highest match to the target.
    """

    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    return max(strings, key=lambda s: similarity(s, target))


def get_namespace(uri: str) -> str:
    """Get the namespace of a full URI."""
    if "#" in uri:
        return uri.rsplit("#", 1)[0]
    else:
        return uri.rsplit("/", 1)[0]


def aggregate_all_resources(resource: dict | list, seen: set):
    """Aggregate all resources (uri) in the resource dict or list."""

    def _add_resource(r: Any):
        """Add resource to the set."""
        if not isinstance(r, str) or not r.startswith("http://"):
            return
        seen.add(r)

    iterator = resource if isinstance(resource, list) else resource.values()
    for r in iterator:
        if isinstance(r, (dict, list)):
            aggregate_all_resources(r, seen)
        else:
            _add_resource(r)


def get_prefix_name(uri: str) -> Tuple[str, str]:
    """Get the prefix of a full URI."""
    if uri.endswith("#") or uri.endswith("/"):
        ns = uri[:-1]
    else:
        if "#" in uri:
            ns = uri.rsplit("#", 1)[0]
        elif "/" in uri:
            ns = uri.rsplit("/", 1)[0]
        else:
            ns = uri

    if ns in INV_PREFIX_MAP:
        prefix = INV_PREFIX_MAP[ns]
    else:
        raise ValueError(f"Cannot find prefix for {uri}")

    n = len(ns)
    name = uri[n:]
    name = name.replace("#", "")
    name = name.replace("/", "")
    return prefix, name


def uri_to_class_name(uri: str, mapping: dict[str, str] | None) -> str:
    """Generate class or slot name from uri.

    Args:
        uri (str): uri of the class or slot
    """
    if mapping is not None and uri in mapping:
        return mapping[uri]
    else:
        prefix, name = get_prefix_name(uri)
        out = prefix.capitalize() + name[0].capitalize() + name[1:]
        if mapping is not None:
            mapping[uri] = out
        return out


def all_value_uri_to_class_name(
    resource: dict | list, mapping: dict[str, str] | None = None
):
    """Convert all uris in the resource to class names."""
    if mapping is None:
        mapping = {}

    iterator = (
        enumerate(resource) if isinstance(resource, list) else resource.items()
    )

    for i, r in iterator:
        if isinstance(r, (dict, list)):
            all_value_uri_to_class_name(r, mapping)
        else:
            if not is_uri(r):
                continue
            if isinstance(i, str) and (i == "class_uri" or i == "slot_uri"):
                continue
            resource[i] = uri_to_class_name(r, mapping)


def is_uri(s: Any) -> bool:
    """Check if a string is a URI."""
    return isinstance(s, str) and s.startswith("http://")


def all_key_uri_to_class_name(
    resource: dict | list,
    mapping: dict[str, str],
):
    """Convert all uri in keys to class names."""
    if isinstance(resource, dict):
        key_to_change = []
        for k, r in resource.items():
            if is_uri(k):
                key_to_change.append(k)
            if isinstance(r, dict):
                all_key_uri_to_class_name(r, mapping)

        for k in key_to_change:
            resource[uri_to_class_name(k, mapping)] = resource.pop(k)
    elif isinstance(resource, list):
        for r in resource:
            if isinstance(r, dict):
                all_key_uri_to_class_name(r, mapping)
    else:
        return


def gen_final_schema_definition(
    import_engine: OwlImportEngine,
    all_classes: dict,
    all_slots: dict,
    local_types: str | list[str] | None = None,
) -> dict:
    """Generate final schema.

    Pooling all aggregated classes and slots.
    """
    import_engine.add_info("classes", "Any", "class_uri", "linkml:Any")
    schema = import_engine.schema
    schema["classes"] = all_classes
    schema["slots"] = all_slots

    # read in any custom types
    if local_types is not None:
        if isinstance(local_types, str):
            local_types = [local_types]
        for t in local_types:
            with open(t) as f:
                _types = yaml.safe_load(f)
            import_engine.types = {
                **import_engine.types,
                **_types.get("types", {}),
            }
        schema["types"] = import_engine.types
    # schema_def = SchemaDefinition(**schema)
    return schema


def merge_info(info_dict_to, info_dict_from, overwrite=False, logger=None):
    """Merge info_dict_from into info_dict_to."""
    for name, info_from in info_dict_from.items():
        if name in info_dict_to:
            info_to = info_dict_to[name]
            if isinstance(info_to, dict) and isinstance(info_from, dict):
                merge_info(info_to, info_from, overwrite, logger)
            else:
                if overwrite:
                    if logger:
                        logger.warning(f"Overwriting {name}")
                    info_dict_to[name] = info_from
        else:
            info_dict_to[name] = info_from


def generate_schema(
    filename: str,
    ofn_dir: str,
    out_dir: str,
    local_imports: str | list[str] | None = None,
    log_level: str = "INFO",
):
    """Generate schema.

    Assuming all ofn files are available and stored in the same directory.

    Args:
        filename (str): The name of the main ontology file.
        ofn_dir (str): The directory containing all ofn files.
        out_dir (str): The directory to save the output schema file.
        local_imports (str | list[str]): local imports for the schema.
        log_level (str): logging level.
    """
    file_queue: deque = deque()
    file_queue.append(filename)
    schema_info = {}
    target_resources: defaultdict = defaultdict(None)

    logger = logging.getLogger(TOOL_NAME)

    def _add_to_target_resources(resources: set[str]):
        """Add resources to target_resources."""
        for resource in resources:
            if resource == "http://purl.org/dc/elements/1.1/relation":
                pass
            ns = get_namespace(resource)
            _fn = INV_PREFIX_MAP[ns] + ".ofn"
            if _fn not in target_resources:
                target_resources[_fn] = set()
            if resource in target_resources[_fn]:
                continue
            logger.debug(f"adding {resource} to {_fn}")
            target_resources[_fn].add(resource)
            if _fn not in file_queue:
                # new resource found, add to queue
                file_queue.append(_fn)

    while file_queue:
        fn = file_queue.popleft()
        logger.info(f"Processing {fn}")

        if fn not in schema_info:
            import_engine = OwlImportEngine(
                local_imports=local_imports, log_level=log_level
            )
            import_engine.convert(str(pathlib.Path(ofn_dir) / fn))
            import_engine.aggregate_resources(target_resources.get(fn, None))
            schema_info[fn] = import_engine
        else:
            import_engine = schema_info[fn]
            import_engine.aggregate_resources(target_resources.get(fn, None))

        _add_to_target_resources(import_engine.resources_this)
        _add_to_target_resources(import_engine.resources_foreign)

    # merge class and slots

    all_classes = deepcopy(schema_info[filename].classes)
    all_slots = deepcopy(schema_info[filename].slots)

    for fn, resource in target_resources.items():
        classes = {}
        slots = {}
        for r in resource:
            if r in schema_info[fn].classes:
                classes[r] = schema_info[fn].classes[r]
            elif r in schema_info[fn].slots:
                slots[r] = schema_info[fn].slots[r]
            else:
                Warning(f"resource {r} not found in schema_info of {fn}")
        merge_info(all_classes, classes, logger=logger, overwrite=False)
        merge_info(all_slots, slots, logger=logger, overwrite=False)

    # verify results
    all_resources: set[str] = set()
    aggregate_all_resources(all_classes, all_resources)
    aggregate_all_resources(all_slots, all_resources)
    uri_to_cls_name_mapping = {}
    for c in all_resources:
        if c in all_classes or c in all_slots:
            uri_to_cls_name_mapping[c] = uri_to_class_name(c, None)
            continue
        else:
            logger.error(f"resource {c} not found in all_classes or all_slots")

    # generate class name from uri
    all_value_uri_to_class_name(all_classes, uri_to_cls_name_mapping)
    all_value_uri_to_class_name(all_slots, uri_to_cls_name_mapping)

    # fix key names
    all_key_uri_to_class_name(all_classes, uri_to_cls_name_mapping)
    all_key_uri_to_class_name(all_slots, uri_to_cls_name_mapping)

    # generate final schema
    final_schema = gen_final_schema_definition(
        schema_info[filename], all_classes, all_slots, local_imports
    )
    stem = filename.split(".")[0]
    out_path = pathlib.Path(out_dir) / f"{stem}.yaml"
    write_schema(final_schema, str(out_path))

    # fix class definition sequence
    fix_class_definition_sequence(str(out_path), overwrite=True)


@dataclass
class OntoNode:
    """Ontology node class."""

    uri: str
    depend_on_nodes: set[OntoNode]  # the set of node that this node depends on
    dependent_nodes: set[OntoNode]  # the set of node that depends on this node

    def __repr__(self):
        """Return the string representation of the node."""
        return self.uri

    def __eq__(self, object):
        """Check if two nodes are equal."""
        return self.uri == object.uri

    def __hash__(self):
        """Return the hash of the node."""
        return hash(self.uri)


class OntoDependencySorter:
    """Class for sorting the dependency between ontologies."""

    def __init__(self, inp_dir: str):
        """Initialize the OntoDependencySorter."""
        self.inp_dir = pathlib.Path(inp_dir)
        if not self.inp_dir.exists():
            raise FileNotFoundError(f"Directory not found: {inp_dir}")
        self.all_nodes: dict[str, OntoNode] = dict()

    def traverse(self, name: str, uri: str):
        """Traverse the ontology graph."""
        # read the source ontology file
        name = EQUIVALENT_PREFIXES.get(name, name)
        uri = EQUIVALENT_URIS.get(uri, uri)
        uri = uri if uri is not None else PREFIX_MAP.get(name, "None")

        name_by_uri = INV_PREFIX_MAP.get(uri, "None")
        name = name_by_uri if name_by_uri != "None" else name

        ofn_file_name = URI_FILE_MAP.get(uri, name + ".ofn")
        if (
            self.inp_dir.joinpath(ofn_file_name).exists()
            and uri not in URI_FILE_MAP
        ):
            URI_FILE_MAP[uri] = ofn_file_name

        print(f"Traversing {name} from {ofn_file_name}, {uri}")
        doc, ontology = read_ofn_file(str(self.inp_dir / ofn_file_name))
        prefix_dict, inv_prefix_dict = extract_prefixes(
            doc, name, rm_trailing_char=True
        )
        print(f"Prefix from {name}: {prefix_dict}")

        if uri == "None":
            uri = prefix_dict.get(name, "None")

        if uri == "None":
            uri = PREFIX_MAP.get(name, "None")

        if uri == "None":
            raise ValueError(f"Cannot find uri for {name}")

        if name not in PREFIX_MAP:
            PREFIX_MAP[name] = uri
            INV_PREFIX_MAP[uri] = name

        # prefix_dict: from prefix to uri, inv_prefix_dict: from uri to prefix

        if uri not in self.all_nodes:
            # create new node
            node = OntoNode(uri, set(), set())
            self.all_nodes[uri] = node
        else:
            node = self.all_nodes[uri]

        for p, p_uri in prefix_dict.items():
            p = EQUIVALENT_PREFIXES.get(p, p)
            p_uri = EQUIVALENT_URIS.get(p_uri, p_uri)
            if p in IGNORE_PREFIXES or p == name:
                continue
            if p_uri in self.all_nodes:  # already added
                other = self.all_nodes[p_uri]
            else:
                # other = OntoNode(p_uri, set(), set())
                # self.all_nodes[p_uri] = other
                self.traverse(p, p_uri)
                other = self.all_nodes[p_uri]
            # node depends on other
            if node not in other.dependent_nodes:
                other.dependent_nodes.add(node)
            # other depends on node
            if other not in node.depend_on_nodes:
                node.depend_on_nodes.add(other)


class OwlImportEngine(ImportEngine):
    """Takes OWL file and converts it to a LinkML schema."""

    FORBIDDEN_CLASS_NAMES = ["List", "Dict", "Set", "Tuple"]
    TYPES_MAP = {t.type_class_curie: t.type_name for t in LINKML_TYPES}

    def __init__(
        self,
        agent_cfg: str | None = None,
        log_level=logging.INFO,
        local_imports: str | list[str] | None = None,
    ):
        """Initialize the OwlImportEngine.

        Args:
            agent_cfg (str): path to the agent configuration file
            log_level (optional): logging level. Defaults to logging.INFO.
            local_imports (str | list[str]): local imports for the schema.
        """
        self.logger = logging.getLogger(TOOL_NAME)
        self.logger.setLevel(log_level)

        self.include_unmapped_annotations = False
        self.classes: Dict[str, Dict] = {}
        self.slots: Dict[str, Dict] = {}
        self.enums: Dict[Any, Any] = {}
        self.types: Dict[Any, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.prefix_dict: Dict[str, str] = {}
        self.inv_prefix_dict: Dict[str, str] = {}
        self.iri_to_name_map: Dict[str, str] = {}
        self.name_to_iri_map: Dict[str, str] = {}
        self.all_properties: Dict[str, str] = {}
        self.same_slots: Dict[str, str] = {}

        self.subclassof: defaultdict[str, set] = defaultdict(set)
        self.subclassof_info: defaultdict[str, list] = defaultdict(list)
        self.slot_isamap: defaultdict[str, set] = defaultdict(set)
        self.slot_usage: defaultdict[str, dict] = defaultdict(dict)
        self.single_valued_slots: Set[str] = set()
        self.self_prefix = "None"
        self.resources_this: Set[str] = set()
        self.resources_foreign: Set[str] = set()

        if local_imports is not None:
            if isinstance(local_imports, str):
                local_imports = [local_imports]
            self.local_imports = local_imports
            for imp in local_imports:
                self.logger.info(f"Adding local import: {imp}")
                self.TYPES_MAP = {
                    **self.TYPES_MAP,
                    **self.get_local_import_types(imp),
                }
        else:
            local_imports = []
        self.check_type_map()

        self.agent_relevant_slot = (
            AgentRelevantSlot(agent_cfg) if agent_cfg is not None else None
        )

        self.agent_assign_slot = (
            AgentAssignSlot(agent_cfg) if agent_cfg is not None else None
        )

    def check_type_map(self):
        """Make sure type map is one to one."""
        inv_map = {}
        for k, v in self.TYPES_MAP.items():
            if v in inv_map:
                self.logger.error(
                    f"Duplicate type {v} for {k} and {inv_map[v]}"
                )
                raise ValueError(
                    f"Duplicate type {v} for {k} and {inv_map[v]}"
                )
            inv_map[v] = k

    def reset(self):
        """Reset the internal state of the engine."""
        for name, value in self.__dict__.items():
            # skip if it is a class attribute
            if name in self.__class__.__dict__:
                self.logger.debug(f"Skipping {name}")
                continue
            self.logger.debug(f"Resetting {name}")
            if isinstance(value, (dict, defaultdict, set)):
                value.clear()

    def iri_as_name(self, v):
        """Convert IRI to name."""
        v = str(v)
        if v in self.iri_to_name_map:
            return self.iri_to_name_map[v]
        n = self._as_name(v)
        if n != v:
            while n in self.name_to_iri_map:
                n = "Alt" + n
            self.name_to_iri_map[n] = v
            self.iri_to_name_map[v] = n
        return n

    def get_name(self, entity):
        """Get the name of an entity."""
        return self.iri_as_name(entity)

    def _as_name(self, v):
        v = str(v)
        for sep in ["#", "/", ":"]:
            if sep in v:
                return v.split(sep)[-1]
        return v

    def read_ofn_file(self, file: str) -> Tuple[OntologyDocument, Ontology]:
        """Read an OWL file (*.ofn) and return the ontology document."""
        return read_ofn_file(file)

    def extract_prefixes(
        self,
        doc: OntologyDocument,
        name: Union[str, None] = None,
        rm_trailing_char=False,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Extract prefixes from an ontology document."""
        prefix_dict, inv_prefix_dict = extract_prefixes(
            doc, name, rm_trailing_char
        )
        if "schema" in prefix_dict:
            schema_iri = prefix_dict.pop("schema")
            inv_prefix_dict.pop(schema_iri)
        return prefix_dict, inv_prefix_dict

    def extract_prefixes_ttl(
        self, ttl: str, name: Union[str, None] = None
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Extract prefixes from a TTL file."""
        # read ttl file
        prefix_dict = {}
        inv_prefix_dict = {}
        g = Graph()
        g = g.parse(ttl, format="turtle")
        for prefix, namespace in g.namespace_manager.namespaces():
            prefix = prefix if prefix else name
            prefix_dict[prefix] = str(namespace)
            inv_prefix_dict[str(namespace)] = prefix
        if "schema" in prefix_dict:
            schema_iri = prefix_dict.pop("schema")
            inv_prefix_dict.pop(schema_iri)
        return prefix_dict, inv_prefix_dict

    def extract_imports(self, ontology: Ontology) -> Dict[str, str]:
        """Extract imports from an ontology."""
        imports_dict = {}
        for i in ontology.directlyImportsDocuments:
            imports_dict[self.get_name(i.iri)] = i.iri.v
        return imports_dict

    def expand_intersection(
        self, c: ObjectIntersectionOf, intersect_list=None
    ):
        """Recursively expand intersections."""
        if intersect_list is None:
            intersect_list = []
        for x in c.classExpressions:
            if not isinstance(x, ObjectIntersectionOf):
                intersect_list.append(x)
            else:
                self.expand_intersection(x, intersect_list)

        return intersect_list

    def expand_unions(self, c: ObjectUnionOf, union_list=None):
        """Recursively expand unions."""
        if union_list is None:
            union_list = []
        for x in c.classExpressions:
            if not isinstance(x, ObjectUnionOf):
                union_list.append(x)
            else:
                self.expand_unions(x, union_list)
        return union_list

    def expand(self, c: Union[ObjectIntersectionOf, ObjectUnionOf], lst=None):
        """Expand class expressions in an intersection or union."""
        if lst is None:
            lst = []
        for x in c.classExpressions:
            if isinstance(x, ObjectIntersectionOf) or isinstance(
                x, ObjectUnionOf
            ):
                self.expand(x, lst)
            else:
                lst.append(x)
        return lst

    def set_slot_usage(self, cls_uri, property, usage_key, usage_value):
        """Set the slot usage for a child class."""
        if "slot_usage" not in self.classes[cls_uri]:
            self.classes[cls_uri]["slot_usage"] = {}
        su = self.classes[cls_uri]["slot_usage"]
        if property not in su:
            su[property] = {}
        su[property][usage_key] = usage_value

    def set_cardinality(self, cls_uri, property, min_card, max_card):
        """Set the cardinality of slot for a child class."""
        if max_card is not None:
            if max_card == 1:
                self.set_slot_usage(cls_uri, property, "multivalued", False)
            elif max_card > 1:
                self.set_slot_usage(cls_uri, property, "multivalued", True)
        if min_card is not None:
            if min_card == 1:
                self.set_slot_usage(cls_uri, property, "required", True)
            elif min_card == 0:
                self.set_slot_usage(cls_uri, property, "required", False)
            else:
                self.set_slot_usage(cls_uri, property, "multivalued", True)

    def process_subclassof(self):
        """Process the subclass relationships."""
        self.logger.info("Processing subclass relationships for classes...")
        for child_cls_uri, subclassof_list in self.subclassof_info.items():
            self._process_subclassof(child_cls_uri, subclassof_list)

    def _process_subclassof(self, child_cls_uri, subclassof_list):
        """Process the inheritance for classes.

        Args:
            subclassof_list: list of subclassof axioms for one child class
        """
        _set_slot_usage = partial(self.set_slot_usage, child_cls_uri)
        _set_cardinality = partial(self.set_cardinality, child_cls_uri)

        def _set_slot_usage_range(e: ClassExpression):
            rng_list = []
            for x in self.expand(e):
                if isinstance(x, Class):
                    rng_list.append(self.to_full_uri(str(x.v)))
                if rng_list:  # remove duplicates
                    rng_list = list(set(rng_list))
                if len(rng_list) == 1:
                    _set_slot_usage(p, "range", rng_list[0])
                else:
                    _set_slot_usage(p, "range", "Any")
                    _set_slot_usage(
                        p, "any_of", [{"range": v} for v in rng_list]
                    )

        subclassof_class_set = set()
        # self.logger.debug(f"Processing parent of {child}")
        for parent in subclassof_list:
            p = None
            # self.logger.debug(f"\tProcessing parent: {parent}")
            # begin switch cases
            if isinstance(parent, Class):
                p = self.to_full_uri(parent)
                subclassof_class_set.add(p)  # add to set
            elif isinstance(parent, DataExactCardinality):
                p = self.to_full_uri(parent.dataPropertyExpression)
                _set_cardinality(p, parent.card, parent.card)
            elif isinstance(parent, ObjectExactCardinality):
                p = self.to_full_uri(parent.objectPropertyExpression)
                _set_cardinality(p, parent.card, parent.card)
            elif isinstance(parent, ObjectMinCardinality):
                p = self.to_full_uri(parent.objectPropertyExpression)
                _set_cardinality(p, parent.min_, None)
            elif isinstance(parent, DataMinCardinality):
                p = self.to_full_uri(parent.dataPropertyExpression)
                _set_cardinality(p, parent.min_, None)
            elif isinstance(parent, ObjectMaxCardinality):
                p = self.to_full_uri(parent.objectPropertyExpression)
                _set_cardinality(p, None, parent.max_)
            elif isinstance(parent, DataMaxCardinality):
                p = self.to_full_uri(parent.dataPropertyExpression)
                _set_cardinality(p, None, parent.max_)
            elif isinstance(parent, ObjectAllValuesFrom):
                p = self.to_full_uri(parent.objectPropertyExpression)
                if isinstance(parent.classExpression, Class):
                    cn = self.to_full_uri(parent.classExpression)
                    _set_slot_usage(p, "range", cn)
                elif isinstance(
                    parent.classExpression, ObjectUnionOf
                ) or isinstance(parent.classExpression, ObjectIntersectionOf):
                    _set_slot_usage_range(parent.classExpression)
                else:
                    self.logger.error(
                        "Cannot yet handle anonymous ranges:"
                        + f" {parent.classExpression}"
                    )
            elif isinstance(parent, ObjectSomeValuesFrom):
                p = self.to_full_uri(parent.objectPropertyExpression)
                _set_cardinality(p, 1, None)
                if isinstance(parent.classExpression, Class):
                    cn = self.to_full_uri(parent.classExpression)
                    _set_slot_usage(p, "range", cn)
                elif isinstance(
                    parent.classExpression, ObjectUnionOf
                ) or isinstance(parent.classExpression, ObjectIntersectionOf):
                    _set_slot_usage_range(parent.classExpression)
                else:
                    self.logger.error(
                        "Cannot yet handle anonymous ranges:"
                        + f" {parent.classExpression}"
                    )
            elif isinstance(parent, DataSomeValuesFrom):
                if len(parent.dataPropertyExpressions) == 1:
                    p = self.to_full_uri(parent.dataPropertyExpressions[0])
                    _set_cardinality(p, 1, None)
                else:
                    self.logger.error(
                        "Cannot handle multiple data property"
                        + f" expressions: {parent}"
                    )
            elif isinstance(parent, DataAllValuesFrom):
                if len(parent.dataPropertyExpressions) == 1:
                    p = self.to_full_uri(parent.dataPropertyExpressions[0])
                    r = parent.dataRange
                    if isinstance(r, DataOneOf):
                        self.logger.error(f"TODO: enum for {r}")
                    elif isinstance(r, Datatype):
                        _set_slot_usage(p, "range", r)
                    else:
                        self.logger.error(f"Cannot handle range of {r}")
                else:
                    self.logger.error(
                        "Cannot handle multiple data property"
                        + f" expressions: {parent}"
                    )
            elif isinstance(parent, DataHasValue):
                p = self.to_full_uri(parent.dataPropertyExpression)
                lit = parent.literal.v
                if isinstance(lit, TypedLiteral):
                    lit = lit.literal
                _set_slot_usage(p, "equals_string", str(lit))
            else:
                self.logger.error(
                    f"cannot handle anonymous parent classes for {parent}",
                )
            # end switch cases

        subclassof_class_list = list(subclassof_class_set)
        if len(subclassof_class_list) == 0:
            return
        p = subclassof_class_list.pop()
        self.add_info("classes", child_cls_uri, "is_a", p)

        for p in subclassof_class_list:
            self.add_info("classes", child_cls_uri, "mixins", p, True)

    def add_class_info(self, *args, **kwargs):
        """Add class information to the schema."""
        self.add_element_info("classes", *args, **kwargs)

    def add_slot_info(self, *args, **kwargs):
        """Add slot information to the schema."""
        self.add_element_info("slots", *args, **kwargs)

    def add_range(self, sn, range_cn):
        """Add range information to the schema."""
        if sn not in self.schema["slots"]:
            self.schema["slots"][sn] = {}
        if "range" not in self.schema["slots"][sn]:
            self.schema["slots"][sn]["range"] = []
        self.schema["slots"][sn]["range"].append(range_cn)

    def add_element_info(
        self, type: str, cn: str, sn: str, v: Any, multivalued=False
    ):
        """Add element information to the schema."""
        if cn not in self.schema[type]:
            self.schema[type][cn] = {}
        c = self.schema[type][cn]
        if multivalued:
            if sn not in c:
                c[sn] = []
            c[sn].append(v)
        else:
            if sn in c and v != c[sn]:
                self.logger.error(f"Overwriting {sn} for {c} to {v}, skipping")
                return
            c[sn] = v

    def process_slot(self):
        """Add the slot information to the schema."""
        self.logger.info("Processing slots...")
        for c, parents in self.slot_isamap.items():
            parents = list(parents)
            p = parents.pop()
            self.add_slot_info(c, "is_a", p)
            for p in parents:
                self.add_slot_info(c, "mixins", p, True)

        for sn, s in self.schema["slots"].items():
            if "range" not in self.schema["slots"][sn]:
                self.add_slot_info(sn, "range", "Any")

            if "domain" not in self.schema["slots"][sn]:
                self.add_slot_info(sn, "domain", "Any")

            if "multivalued" not in s:
                s["multivalued"] = sn not in self.single_valued_slots
            if "range" in s:
                if isinstance(s["range"], list):
                    rg = s["range"]
                    if len(rg) == 0:
                        del s["range"]
                    elif len(rg) == 1:
                        s["range"] = rg[0]
                    else:
                        del s["range"]
                        s["any_of"] = [{"range": x} for x in rg]

    def process_annotation_assertion(self, axiom: Axiom):
        """Process annotation assertions.

        Simplified version.
        """
        if not isinstance(axiom, AnnotationAssertion):
            return
        p = axiom.property
        pstr = str(p)
        sub = self.to_full_uri(axiom.subject.v)
        if sub not in self.classes and sub not in self.slots:
            return
        val = axiom.value.v
        if isinstance(val.v, StringLiteralWithLanguage):
            content = val.v.literal
        else:
            content = val.v

        def _cls_or_slot(sub):
            if sub in self.classes:
                return "classes"
            elif sub in self.slots:
                return "slots"
            else:
                raise ValueError(f"{sub} is not in classes or slots.")

        if pstr == "rdfs:comment":
            self.add_info(_cls_or_slot(sub), sub, "comments", content, True)
        elif pstr.endswith(":description"):
            self.add_info(
                _cls_or_slot(sub), sub, "description", content, False
            )
        if pstr == "rdfs:label":
            self.add_info(_cls_or_slot(sub), sub, "title", content, False)

    def _process_annotation_assertion(self, axiom: Axiom):
        """Process annotation assertions.

        Original code.
        """
        if not isinstance(axiom, AnnotationAssertion):
            return
        p = axiom.property
        strp = str(p)
        sub = axiom.subject.v
        val = axiom.value.v
        if isinstance(sub, IRI):
            sub = self.get_name(sub)
            if isinstance(val, AnonymousIndividual):
                self.logger.info(
                    f"Ignoring anonymous individuals: {sub} {strp} {val}"
                )
                return
            elif isinstance(val, Literal):
                val = str(val.v)
            elif isinstance(val, IRI):
                val = val.v
            else:
                val = str(val)
            if sub in self.classes:
                t = "classes"
            elif sub in self.slots:
                t = "slots"
            else:
                self.logger.error(f"{sub} is not known")
                return
            if t is not None:
                if strp == "rdfs:comment":
                    self.add_element_info(
                        t, sub, "description", val, multivalued=False
                    )
                elif strp == "schema:rangeIncludes":
                    range_cn = self.get_name(val)
                    self.logger.error(
                        f"UNTESTED RANGE: schema.org {sub} {val} // {range_cn}"
                    )
                    self.add_range(sub, range_cn)
                elif strp == "schema:domainIncludes":
                    domain_cn = self.get_name(val)
                    self.logger.error(
                        f"UNTESTED: schema.org {sub} {val} // {domain_cn}"
                    )
                    if domain_cn not in self.schema["classes"]:
                        self.schema["classes"][domain_cn] = {}
                    if "slots" not in self.schema["classes"][domain_cn]:
                        self.schema["classes"][domain_cn]["slots"] = []
                    self.schema["classes"][domain_cn]["slots"].append(sub)
                else:
                    if self.include_unmapped_annotations:
                        self.add_element_info(
                            t,
                            sub,
                            "comments",
                            f"{p} = {val}",
                            multivalued=True,
                        )

    def process_slot_usage(self, slots):
        """Process the slot usage."""
        self.logger.info("Processing slot usage...")
        for cn, usage in self.slot_usage.items():
            # example usage content:
            # {'contains': {'range':'Text', 'domain':'Any',
            # 'required': True, 'multivalued': True}}
            _usage = {}
            for sn, u in usage.items():
                if sn not in slots:
                    # check first if it has an equivalent slot
                    sn_equiv = self.same_slots.get(sn, sn)
                else:
                    sn_equiv = sn

                if sn_equiv not in slots:
                    # retrieve from original source
                    slots[sn_equiv] = self.schema["slots"][sn_equiv]

                if "slots" not in self.schema["classes"][cn]:
                    # initiate the slots list
                    self.schema["classes"][cn]["slots"] = [sn_equiv]
                else:
                    if sn not in self.schema["classes"][cn]["slots"]:
                        # add the slot to the class if it is not already there
                        self.schema["classes"][cn]["slots"].append(sn_equiv)
                _usage[sn_equiv] = u
            self.schema["classes"][cn]["slot_usage"] = _usage

    def add_identifier(self, identifier: str | None):
        """Add an identifier to the schema."""
        if identifier is not None:
            self.slots[identifier] = {
                "identifier": True,
                "range": "uriorcurie",
            }
            for c in self.classes.values():
                if not c.get("is_a", None) and not c.get("mixins", []):
                    if "slots" not in c:
                        c["slots"] = []
                    c["slots"].append(identifier)

    def determine_slot_relevance(self, slots, agent):
        """Determine the relevance of slots to the ontology as a whole.

        Use agenic AI to determine the relevance of slots to classes.
        """

    def compare_slot(self, sn_info_1, sn_info_2):
        """Compare two slot information dictionaries."""
        keys_to_compare = ["range", "domain", "multivalued"]
        for k in keys_to_compare:
            v1 = sn_info_1.get(k, None)
            v2 = sn_info_2.get(k, None)
            if v1 != v2:
                return False
        return True

    def del_by_name(self, slots: dict, del_list: list | set):
        """Delete slots by name."""
        for sn in del_list:
            if sn not in slots:
                continue
            slots.pop(sn)

    def remove_redundant_slots(self, slots: dict):
        """Remove redundant slots from the schema."""
        self.logger.info("Removing redundant slots...")
        # first trim the slots that are the same as another slot
        to_del = set()
        for sn, slot_info in slots.items():
            if "is_a" in slot_info and "mixins" not in slot_info:
                sn_other = slot_info["is_a"]
                if sn_other not in slots:
                    continue
                if self.compare_slot(slot_info, slots[sn_other]):
                    self.same_slots[
                        sn_other
                    ] = sn  # record the equivalent slot for the ones to be
                    # deleted
                    to_del.add(sn_other)
        for sn in to_del:
            self.logger.warning(f"Removing redundant slot {sn}")
        self.del_by_name(slots, to_del)

    def remove_abstract_slots(self, slots: dict):
        """Remove abstract slots from the schema."""
        self.logger.info("Removing abstract slots...")
        abstract_list = ["topObjectProperty", "topDataProperty"]
        to_del = set()
        for sn in slots:
            if sn in abstract_list:
                to_del.add(sn)
        for sn in to_del:
            self.logger.warning(f"Removing abstract slot {sn}")
        self.del_by_name(slots, to_del)

    def remove_irrelevant_slots(self, slots: dict):
        """Remove irrelevant slots from the schema."""
        self.logger.info("Removing irrelevant slots...")
        if len(slots) > 0 and self.agent_relevant_slot is not None:
            # query relevant-slot agent to determine slot relevance, remove
            # irrelevant slots
            relevant_slots = self.agent_relevant_slot.query(slots)
            del_list = []
            for sn, slot_info in slots.items():
                if not relevant_slots[sn]:
                    del_list.append(sn)
            self.del_by_name(slots, del_list)

    def query_slot_assignment(self, slots):
        """Query the slot assignment agent to determine slot assignment."""
        self.logger.info(
            "Querying slot assignment agent (might take a while)..."
        )
        if len(slots) > 0 and self.agent_assign_slot is not None:
            cls_name = self.get_class_names()
            slot_name = self.get_slot_names(slots)
            assign_result = self.agent_assign_slot.query(cls_name, slot_name)
            return assign_result
        else:
            return {}

    def add_slot_to_class(self):
        """Add slots to classes.

        If a class has a slot_usage, the slot will be added to the class.
        """
        self.logger.info("Adding slots to classes...")
        for cn, class_info in self.classes.items():
            if cn in self.slots or cn in self.all_properties:
                # skip classes that are slots
                continue
            if "abstract" in class_info and class_info["abstract"]:
                # skip abstract classes, e.g., linkml:Any
                continue

            if "slot_usage" in class_info:
                for sn, usage in class_info["slot_usage"].items():
                    if "slots" not in class_info:
                        class_info["slots"] = []
                    if sn not in class_info["slots"]:
                        self.logger.debug(f"Adding {sn} to {cn}")
                        class_info["slots"].append(sn)

    def get_class_names(self) -> List[str]:
        """Gather class names in a list.

        Returns:
            list: list of class names
        """
        cls_name = []
        for k, v in self.schema["classes"].items():
            if "abstract" in v and v["abstract"]:
                continue
            if k in self.schema["slots"] or k in self.all_properties:
                continue
            cls_name.append(k)
        return cls_name

    def get_slot_names(self, slots=None) -> List[str]:
        """Return slot names in a list."""
        if slots is None:
            slots = self.schema["slots"]
        return list(slots.keys())

    def rename_forbidden_class_names(self):
        """Rename class if the name coincide with the forbidden class names."""
        self.logger.info("Checking class names...")

        def _rename(old_name):
            return self.schema["name"].capitalize() + old_name

        classes = {}
        for cn, cls_info in self.schema["classes"].items():
            # check "is_a" and "mixinx"
            if "is_a" in cls_info:
                if cls_info["is_a"] in self.FORBIDDEN_CLASS_NAMES:
                    cls_info["is_a"] = _rename(cls_info["is_a"])
            if "mixins" in cls_info:
                for i, mixin in enumerate(cls_info["mixins"]):
                    if mixin in self.FORBIDDEN_CLASS_NAMES:
                        cls_info["mixins"][i] = _rename(mixin)
            if "slot_usage" in cls_info:
                for sn, usage in cls_info["slot_usage"].items():
                    if (
                        "range" in usage
                        and usage["range"] in self.FORBIDDEN_CLASS_NAMES
                    ):
                        usage["range"] = _rename(usage["range"])
                    if "any_of" in usage:
                        for x in usage["any_of"]:
                            if (
                                "range" in x
                                and x["range"] in self.FORBIDDEN_CLASS_NAMES
                            ):
                                x["range"] = _rename(x["range"])
            if cn in self.FORBIDDEN_CLASS_NAMES:
                new_cn = _rename(cn)
                self.logger.warning(f"Renaming class {cn} to {new_cn}")
                classes[new_cn] = self.schema["classes"][cn]
            else:
                classes[cn] = cls_info

        self.schema["classes"] = classes

    def _process_data_property_range(self, axiom: DataPropertyRange):
        """Process data property range."""

        def _exist_in_map(typename):
            """Check if type exists in the types map."""
            if typename not in self.TYPES_MAP:
                self.logger.warning(f"Unknown type {typename}")
                return False
            return True

        p = axiom.dataPropertyExpression.v
        slot_uri = self.get_name(p)  # name of the property
        data_range = axiom.dataRange  # the range the property
        if isinstance(data_range, Datatype):
            dtype_name = str(data_range.v)
            if _exist_in_map(dtype_name):
                self._add_range_to_slot(slot_uri, self.TYPES_MAP[dtype_name])
            else:
                # will create a class in this case
                self._add_range_to_slot(slot_uri, self.to_full_uri(dtype_name))
        elif isinstance(data_range, DataUnionOf):
            for dtype in data_range.dataRanges:
                dtype_name = str(dtype.v)
                if _exist_in_map(dtype_name):
                    self._add_range_to_slot(
                        slot_uri, self.TYPES_MAP[dtype_name]
                    )
                else:
                    # add as class
                    self._add_range_to_slot(
                        slot_uri, self.to_full_uri(dtype_name)
                    )

        elif isinstance(data_range, DatatypeRestriction):
            dtype_name = str(data_range.datatype.v)
            if _exist_in_map(dtype_name):
                self._add_range_to_slot(slot_uri, self.TYPES_MAP[dtype_name])
            else:
                self._add_range_to_slot(slot_uri, self.to_full_uri(dtype_name))
            for r in data_range.restrictions:
                if r.constrainingFacet.v == "xsd:pattern":
                    self.add_info(
                        "slots", slot_uri, "pattern", r.restrictionValue.v
                    )
                elif r.constrainingFacet.v == "xsd:minInclusive":
                    self.add_info(
                        "slots",
                        slot_uri,
                        "minimum_value",
                        eval(r.restrictionValue.v),
                    )
                elif r.constrainingFacet.v == "xsd:maxInclusive":
                    self.add_info(
                        "slots",
                        slot_uri,
                        "maximum_value",
                        eval(r.restrictionValue.v),
                    )
                else:
                    self.logger.error(f"Cannot handle {r} yet.")
        else:
            self.logger.error(
                f"Cannot handle {data_range} for {slot_uri} yet."
            )

    def get_local_import_types(self, file_path: str):
        """Import local type definitions from a schema file."""
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        assert path.suffix == ".yaml", "Only YAML files are supported."
        # read yaml file
        with open(path) as file:
            schema = yaml.safe_load(file)
        types = schema.get("types", {})
        type_map = {}
        for type_name, type_info in types.items():
            type_map[type_info["uri"]] = type_name
        return type_map

    def process_property_domain(
        self,
        axiom: ObjectPropertyDomain
        | DataPropertyDomain
        | AnnotationPropertyDomain,
    ):
        """Process property domain."""
        if not isinstance(
            axiom,
            (
                ObjectPropertyDomain,
                DataPropertyDomain,
                AnnotationPropertyDomain,
            ),
        ):
            return
        if isinstance(axiom, ObjectPropertyDomain):
            slot_uri = self.to_full_uri(axiom.objectPropertyExpression.v)
        elif isinstance(axiom, DataPropertyDomain):
            slot_uri = self.to_full_uri(axiom.dataPropertyExpression.v)
        elif isinstance(axiom, AnnotationPropertyDomain):
            slot_uri = self.to_full_uri(axiom.property.v)
        domain_expression = (
            axiom.classExpression
            if isinstance(axiom, (ObjectPropertyDomain, DataPropertyDomain))
            else axiom.domain
        )
        if isinstance(domain_expression, Class):
            cls_uri = self.to_full_uri(domain_expression)
            self.add_info("classes", cls_uri, "slots", slot_uri, True)
        if isinstance(domain_expression, ObjectUnionOf):
            for d in domain_expression.classExpressions:
                if isinstance(d, Class):
                    cls_uri = self.get_name(d)
                    self.add_info("classes", cls_uri, "slots", slot_uri, True)

    def process_property_range(
        self,
        axiom: Axiom,
    ):
        """Process property range."""
        if isinstance(axiom, DataPropertyRange):
            self._process_data_property_range(axiom)
        elif isinstance(axiom, ObjectPropertyRange):
            self._process_object_property_range(axiom)
        elif isinstance(axiom, AnnotationPropertyRange):
            self._process_annotation_property_range(axiom)
        else:
            return

    def _add_range_to_slot(self, slot_uri: str, range_uri: str):
        """Add range to slot."""
        if slot_uri in self.slots and "range" in self.slots[slot_uri]:
            if "any_of" in self.slots[slot_uri]:
                # The slot has multiple ranges defined. Add the new range to
                # the list of ranges
                self.slots[slot_uri]["any_of"].append({"range": range_uri})
            else:
                # already has a range defined. Retrieve the existing range and
                # convert it to a list of ranges
                r = self.slots[slot_uri]["range"]
                self.slots[slot_uri]["range"] = "Any"
                self.slots[slot_uri]["any_of"] = [
                    {"range": r},
                    {"range": range_uri},
                ]
        else:
            self.add_info(
                "slots",
                slot_uri,
                "range",
                range_uri,
            )

    def _process_annotation_property_range(
        self, axiom: AnnotationPropertyRange
    ):
        """Process annotation property range."""
        property_uri = self.to_full_uri(axiom.property)
        range_uri = self.to_full_uri(axiom.range)
        self._add_range_to_slot(property_uri, range_uri)

    def _process_object_property_range(self, axiom: ObjectPropertyRange):
        """Process object property range."""
        slot_uri = self.to_full_uri(axiom.objectPropertyExpression.v)
        range_expression = axiom.classExpression
        if isinstance(range_expression, Class):
            self._add_range_to_slot(
                slot_uri, self.to_full_uri(range_expression)
            )
        elif isinstance(range_expression, ObjectUnionOf):
            for r in range_expression.classExpressions:
                if isinstance(r, Class):
                    self._add_range_to_slot(slot_uri, self.to_full_uri(r))
                else:
                    self.logger.error(
                        f"Cannot handle {range_expression} of {slot_uri} yet."
                    )

    def _process_sub_data_property_of(self, axiom: SubDataPropertyOf):
        """Process sub data property of."""
        sub = axiom.subDataPropertyExpression.v
        if isinstance(sub, DataProperty):
            child = self.get_name(sub)
            sup = axiom.superDataPropertyExpression.v
            if isinstance(sup, DataProperty):
                parent = self.get_name(sup)
                self.slot_isamap[child].add(parent)
            else:
                self.logger.error(
                    "cannot handle anonymous data parent properties"
                    + f" for {axiom}"
                )
        else:
            self.logger.error(
                f"cannot handle anonymous data child properties for {axiom}"
            )

    def process_sub_object_property_of(self, axiom: SubObjectPropertyOf):
        """Process sub object property of."""
        if not isinstance(axiom, SubObjectPropertyOf):
            return
        sub = axiom.subObjectPropertyExpression.v
        if isinstance(sub, ObjectPropertyExpression) and isinstance(
            sub.v, ObjectProperty
        ):
            child = self.get_name(sub.v)
            sup = axiom.superObjectPropertyExpression.v
            if isinstance(sup, ObjectPropertyExpression) and isinstance(
                sup.v, ObjectProperty
            ):
                parent = self.get_name(sup.v)
                self.slot_isamap[child].add(parent)
            else:
                self.logger.error(
                    "cannot handle anonymous object parent"
                    + f" properties for {axiom}"
                )
        else:
            self.logger.error(
                f"cannot handle anonymous object child properties for {axiom}"
            )

    def process_sub_property_of(
        self,
        axiom: SubDataPropertyOf
        | SubObjectPropertyOf
        | SubAnnotationPropertyOf,
    ):
        """Process sub property of."""
        if isinstance(axiom, SubDataPropertyOf):
            child_cls_uri = self.to_full_uri(axiom.subDataPropertyExpression.v)
            parent_cls_uri = self.to_full_uri(
                axiom.superDataPropertyExpression.v
            )
        elif isinstance(axiom, SubObjectPropertyOf):
            child_cls_uri = self.to_full_uri(
                axiom.subObjectPropertyExpression.v
            )
            parent_cls_uri = self.to_full_uri(
                axiom.superObjectPropertyExpression.v
            )
        elif isinstance(axiom, SubAnnotationPropertyOf):
            child_cls_uri = self.to_full_uri(axiom.sub.v)
            parent_cls_uri = self.to_full_uri(axiom.super.v)
        else:
            return
        if child_cls_uri not in self.slots:
            # could be an annotation property, which is not in slots
            return
        if "is_a" not in self.slots[child_cls_uri]:
            self.add_info("slots", child_cls_uri, "is_a", parent_cls_uri)
        else:
            self.add_info(
                "slots", child_cls_uri, "mixins", parent_cls_uri, True
            )

    def to_curie(self, uri: str) -> str:
        """Convert to a CURIE."""
        if ":" in uri:
            if uri.startswith(":"):
                return f"{self.self_prefix}{uri}"
            else:
                return uri
        else:
            name = self._as_name(uri)
            prefix_uri = self.find_best_match(
                list(self.inv_prefix_dict.keys()), uri
            )
            prefix = self.inv_prefix_dict[prefix_uri]
            return f"{prefix}:{name}"

    def get_prefix_uri(self, prefix: str) -> str:
        """Get the prefix URI."""
        prefix_uri = self.prefix_dict[prefix]
        if not prefix_uri.endswith("/") and not prefix_uri.endswith("#"):
            prefix_uri += "/"
        return prefix_uri

    def to_full_uri(self, uri: str | Any) -> str:
        """Convert to a URI.

        Returns as a tuple with prefix uri and name
        """
        if not isinstance(uri, str):
            uri = str(uri)
        if ":" in uri and not uri.startswith("http:"):
            if uri.startswith(":"):
                return self.get_prefix_uri(self.self_prefix) + uri[1:]
            else:
                prefix, name = uri.split(":")
                return self.get_prefix_uri(prefix) + name
        else:
            return uri

    def add_info(self, type, uri, slot_name, slot_value, multi_valued=False):
        """Add slot information to the class."""
        if type == "classes":
            _dict = self.classes
        elif type == "slots":
            _dict = self.slots
        else:
            raise ValueError(f"Unknown type {type}")
        if uri not in _dict:
            _dict[uri] = {}
        cls = _dict[uri]
        if multi_valued:
            if slot_name not in cls:
                cls[slot_name] = []
            cls[slot_name].append(slot_value)
        else:
            if slot_name in cls and slot_value != cls[slot_name]:
                self.logger.error(
                    f"Overwriting {slot_name} for {cls} to {slot_value},"
                    "skipping"
                )
                return
            cls[slot_name] = slot_value

    def process_declaration(self, axiom):
        """Process declarations."""
        if not isinstance(axiom, Declaration):
            return
        e: IRI = axiom.v
        declaration_type = e.__class__.__name__
        if declaration_type == "NamedIndividual":
            self.logger.debug(f"Ignoring {e}, a NamedIndividual")
            return
        full_uri = self.to_full_uri(str(e.v))
        if declaration_type == "Class":
            self.add_info("classes", full_uri, "class_uri", full_uri)
            self.logger.debug(f"Adding {full_uri} to classes.")
        elif declaration_type in (
            "ObjectProperty",
            "DataProperty",
            "AnnotationProperty",
        ):
            self.add_info("slots", full_uri, "slot_uri", full_uri)
            self.logger.debug(f"Adding {full_uri} to slots.")
        else:
            self.logger.debug(f"Ignoring {full_uri}, a {declaration_type}")

    def _aggregate_resources(
        self,
        resource: dict | list,
        queue: list,
        seen: set,
    ):
        """Add all classes and slots to resources.

        Differentiate between resources of the current namespace and those of
        other namespaces.
        """

        def _add_resource(r: Any):
            """Add resource to the set."""
            if not isinstance(r, str) or not r.startswith("http://"):
                # print("Ignoring", r)
                return
            self.logger.debug(f"Adding {r} to resources")
            queue.append(r)  # only add uri (key) to the queue

        if isinstance(resource, list):
            for r in resource:
                if isinstance(r, (dict, list)):
                    self._aggregate_resources(r, queue, seen)
                else:
                    _add_resource(r)
        elif isinstance(resource, dict):
            for k, r in resource.items():
                # if k in seen:
                #     continue
                if isinstance(r, (dict, list)):
                    self._aggregate_resources(r, queue, seen)
                else:
                    _add_resource(r)

    def gather_subclassof_definition(self, axiom: Axiom):
        """Gather subclass of definitions.

        The SubClassOf definitions of a single child class can span multiple
        axioms. This function gathers all the subclassof definitions for a
        single child class and store them in a dictionary.
        """
        if not isinstance(axiom, SubClassOf):
            return

        if not isinstance(axiom.subClassExpression, Class):
            self.logger.error(
                f"cannot handle anonymous child classes for {axiom}"
            )
            return

        # SubClassOf(Class, Some parent class)
        # get full uri of the child class
        child_cls_uri = self.to_full_uri(str(axiom.subClassExpression.v))

        if isinstance(
            axiom.superClassExpression, ObjectIntersectionOf
        ) or isinstance(axiom.superClassExpression, ObjectUnionOf):
            self.subclassof_info[child_cls_uri] += self.expand(
                axiom.superClassExpression
            )
        else:
            self.subclassof_info[child_cls_uri].append(
                axiom.superClassExpression
            )

    def aggregate_resources(
        self, target_resources: list[str] | set[str] | None = None
    ):
        """Aggregate resources by target."""
        if target_resources is not None:
            # aggregate only target resources
            classes = {
                k: v for k, v in self.classes.items() if k in target_resources
            }
            slots = {
                k: v for k, v in self.slots.items() if k in target_resources
            }
        else:
            classes = self.classes
            slots = self.slots

        queue: list[str] = []
        for k in classes.keys():
            queue.append(k)
        for k in slots.keys():
            queue.append(k)
        seen: set[str] = set()
        while len(queue) > 0:
            k = queue.pop()
            if k in seen:
                continue
            seen.add(k)
            if k not in self.classes and k not in self.slots:
                # it is a resource from another namespace potentially
                continue
            r_dict = self.classes[k] if k in self.classes else self.slots[k]
            self._aggregate_resources(r_dict, queue, seen)

        for r in seen:
            if r.startswith(self.schema["id"]):
                if r not in self.resources_this:
                    self.resources_this.add(r)
            else:
                if r not in self.resources_foreign:
                    self.resources_foreign.add(r)

    def ttl_to_ofn(self, filepath: pathlib.Path) -> pathlib.Path:
        """Convert ttl specified by filepath to ofn format.

        Args:
            filepath (pathlib.Path): path to the ttl file.

        Returns:
            pathlib.Path: path to the ofn file.
        """
        # execute system command
        ofn_dir = filepath.parent / ".ofn"
        ofn_dir.mkdir(exist_ok=True)
        ofn_filepath = ofn_dir / filepath.name.replace(".ttl", ".ofn")
        if not ofn_filepath.exists():
            self.logger.info(f"Converting {filepath} to .ofn using ROBOT")
            if os.system(
                f"robot convert -i {str(filepath)} "
                + f"-output {str(ofn_filepath)}"
            ):
                raise Exception("Error converting to .ofn")
        return ofn_filepath

    def convert(
        self,
        file: str,
        name: str | None = None,
        identifier: str | None = None,
    ) -> SchemaDefinition:
        """
        Convert an OWL schema-style ontology.

        Args:
            file (str): path to the OWL file
            name (str): name of the schema
            identifier (str): identifier for the schema
        """
        self.reset()

        filepath = pathlib.Path(file)
        # check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {file}")

        # check filepath extension
        if filepath.suffix == ".ttl":
            ofn_filepath = self.ttl_to_ofn(filepath)
        elif filepath.suffix != ".ofn":
            raise Exception(f"Invalid file extension: {filepath.suffix}")
        else:
            ofn_filepath = filepath

        doc, ontology = self.read_ofn_file(str(ofn_filepath))

        if name is None:
            name = self.get_name(ontology.iri)
        if name == "None":
            name = ofn_filepath.stem
            ontology.iri = PREFIX_MAP[name]
        self.self_prefix = name

        self.prefix_dict, self.inv_prefix_dict = self.extract_prefixes(
            doc, name, rm_trailing_char=False
        )

        # id is just the namespace of the ontology
        id = (
            ontology.iri
            if ontology.iri
            else PREFIX_MAP.get(self.self_prefix, None)
        )
        if id is None:
            raise ValueError(f"Cannot find id for {self.self_prefix}")

        self.schema = {
            "id": f"{ontology.iri}",
            "name": name,
            "description": name,
            "imports": ["linkml:types"],
            "prefixes": {
                "linkml": "https://w3id.org/linkml/",
                **self.prefix_dict,
            },
            "default_prefix": name,
            "types": self.types,
            "classes": self.classes,
            "slots": self.slots,
            "enums": self.enums,
        }

        # iterate over the axioms and process them
        self.logger.info(f"Processing axioms in {file}...")
        for axiom in ontology.axioms:
            self.logger.debug(f"Axiom: {axiom}")

            self.process_declaration(axiom)
            self.gather_subclassof_definition(axiom)

            # process the SubObjectPropertyOf declarations
            # https://github.com/hsolbrig/funowl/issues/19
            # process SubDataPropertyOf, SubObjectPropertyOf, and
            #  SubAnnotationPropertyOf
            self.process_sub_property_of(axiom)

            # process the domain and range declarations
            # domains become slot declarations
            self.process_property_domain(axiom)

            # process the range declarations of ObjectProperty, DataProperty,
            # and AnnotationProperty
            self.process_property_range(axiom)

            # process annotation assertions, e.g., rdfs:comment, rdfs:label
            self.process_annotation_assertion(axiom)

        # process the parent classes and extract the slot information
        self.process_subclassof()

        # add slots to classes if a class has slot_usage defined
        self.add_slot_to_class()


def fix_class_definition_sequence(
    yaml_file: str, overwrite: bool = True
) -> Union[dict, None]:
    """Fix the sequence of class definitions in LinkML schema YAML file.

    Make sure the class definitions follow the inheritance hierarchy.

    Args:
        yaml_file (str): path to the YAML file
        overwrite (bool, optional): whether to overwrite the file.
                                    Defaults to True.
                                    if set to False, the fixed schema is
                                    returned.
    """
    # read yaml file
    # first check if the file is a yaml file and if the file exists
    logger = logging.getLogger("fix_class_order")
    if not yaml_file.endswith(".yaml"):
        raise ValueError("File is not a YAML file.")
    if not os.path.exists(yaml_file):
        raise FileNotFoundError("File does not exist.")
    with open(yaml_file) as f:
        schema = yaml.safe_load(f)

    if "classes" not in schema:
        raise ValueError("No class definitions")
    classes = deepcopy(schema["classes"])
    classes_ordered = OrderedDict()
    stack: deque[Tuple[str, dict]] = deque()
    while len(classes) > 0:
        # pop from beginning
        c = (k := next(iter(classes)), classes.pop(k))
        stack.append(c)
        while len(stack) > 0:
            c_top = stack[-1]  # get the top of the stack
            cn, c_info = c_top
            if cn in classes_ordered:
                stack.pop()
                continue
            if "is_a" not in c_info:  # no parent
                classes_ordered[cn] = c_info
                stack.pop()
            else:
                parents = [c_info["is_a"]] + c_info.get("mixins", [])
                if all([p in classes_ordered for p in parents]):
                    classes_ordered[cn] = c_info
                    stack.pop()
                else:
                    for parent in parents:
                        if parent in classes_ordered:
                            continue
                        if parent not in classes:
                            raise ValueError(f"Parent {parent} not found")
                        stack.append((parent, classes[parent]))

    schema["classes"] = dict(classes_ordered)
    for k in classes.keys():
        if k not in classes_ordered:
            logger.error(f"Class {k} not ordered")

    if overwrite:
        with open(yaml_file, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
        return None
    else:
        return schema
