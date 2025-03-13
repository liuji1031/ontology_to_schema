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

import logging
import os
import pathlib
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Set, Tuple, Union

import yaml
from funowl import (
    IRI,
    AnnotationAssertion,
    AnnotationProperty,
    AnnotationPropertyRange,
    AnonymousIndividual,
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

from ontology_to_schema.agent_assign_slot import AgentAssignSlot
from ontology_to_schema.agent_relevant_slot import AgentRelevantSlot

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


class OwlImportEngine(ImportEngine):
    """Takes OWL file and converts it to a LinkML schema."""

    FORBIDDEN_CLASS_NAMES = ["List", "Dict", "Set", "Tuple"]

    def __init__(self, agent_cfg: str | None, log_level=logging.INFO):
        """Initialize the OwlImportEngine.

        Args:
            agent_cfg (str): path to the agent configuration file
            log_level (optional): logging level. Defaults to logging.INFO.
        """
        self.logger = logging.getLogger("owl_import_engine")
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
        self.subclassof_list: defaultdict[str, list] = defaultdict(list)
        self.slot_isamap: defaultdict[str, set] = defaultdict(set)
        self.slot_usage_map: defaultdict[str, dict] = defaultdict(dict)
        self.single_valued_slots: Set[str] = set()

        self.agent_relevant_slot = (
            AgentRelevantSlot(agent_cfg) if agent_cfg is not None else None
        )

        self.agent_assign_slot = (
            AgentAssignSlot(agent_cfg) if agent_cfg is not None else None
        )

    def reset(self):
        """Reset the internal state of the engine."""
        for name, value in self.__dict__.items():
            self.logger.debug(f"Resetting {name}")
            if isinstance(value, (dict, defaultdict)):
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
        self, doc: OntologyDocument, name: Union[str, None] = None
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Extract prefixes from an ontology document."""
        prefix_dict = {}
        for p in doc.prefixDeclarations.as_prefixes():
            prefix_name = (
                p.prefixName if len(p.prefixName) > 0 or name is None else name
            )
            prefix_dict[prefix_name] = p.fullIRI

        # also add the imports as prefix
        prefix_dict = {**prefix_dict, **self.extract_imports(doc.ontology)}
        inv_prefix_dict = {v: k for k, v in prefix_dict.items()}
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

    def set_slot_usage(self, child, p, k, v):
        """Set the slot usage for a child class."""
        if p not in self.slot_usage_map[child]:
            self.slot_usage_map[child][p] = {}
        self.slot_usage_map[child][p][k] = v

    def set_cardinality(self, child, p, min_card, max_card):
        """Set the cardinality of slot for a child class."""
        if max_card is not None:
            if max_card == 1:
                self.set_slot_usage(child, p, "multivalued", False)
            elif max_card > 1:
                self.set_slot_usage(child, p, "multivalued", True)
        if min_card is not None:
            if min_card == 1:
                self.set_slot_usage(child, p, "required", True)
            elif min_card == 0:
                self.set_slot_usage(child, p, "required", False)
            else:
                self.set_slot_usage(child, p, "multivalued", True)

    def process_subclassof(self):
        """Process the inheritance for classes."""
        self.logger.info("Processing classes...")
        for child, parents in self.subclassof_list.items():
            _set_slot_usage = partial(self.set_slot_usage, child)
            _set_cardinality = partial(self.set_cardinality, child)

            def _set_slot_usage_range(e: ClassExpression):
                rng_list = []
                for x in self.expand(e):
                    if isinstance(x, Class):
                        rng_list.append(self.get_name(x))
                    if rng_list:
                        rng_list = list(set(rng_list))
                    if len(rng_list) == 1:
                        _set_slot_usage(p, "range", rng_list[0])
                    else:
                        _set_slot_usage(p, "range", "Any")
                        _set_slot_usage(
                            p, "any_of", [{"range": v} for v in rng_list]
                        )

            # self.logger.debug(f"Processing parent of {child}")
            for parent in parents:
                p = None
                # self.logger.debug(f"\tProcessing parent: {parent}")
                # begin switch cases
                if isinstance(parent, Class):
                    p = self.get_name(parent)
                    self.subclassof[child].add(p)  # add to set
                elif isinstance(parent, DataExactCardinality):
                    p = self.get_name(parent.dataPropertyExpression)
                    _set_cardinality(p, parent.card, parent.card)
                elif isinstance(parent, ObjectExactCardinality):
                    p = self.get_name(parent.objectPropertyExpression)
                    _set_cardinality(p, parent.card, parent.card)
                elif isinstance(parent, ObjectMinCardinality):
                    p = self.get_name(parent.objectPropertyExpression)
                    _set_cardinality(p, parent.min_, None)
                elif isinstance(parent, DataMinCardinality):
                    p = self.get_name(parent.dataPropertyExpression)
                    _set_cardinality(p, parent.min_, None)
                elif isinstance(parent, ObjectMaxCardinality):
                    p = self.get_name(parent.objectPropertyExpression)
                    _set_cardinality(p, None, parent.max_)
                elif isinstance(parent, DataMaxCardinality):
                    p = self.get_name(parent.dataPropertyExpression)
                    _set_cardinality(p, None, parent.max_)
                elif isinstance(parent, ObjectAllValuesFrom):
                    p = self.get_name(parent.objectPropertyExpression)
                    if isinstance(parent.classExpression, Class):
                        cn = self.get_name(parent.classExpression)
                        _set_slot_usage(p, "range", cn)
                    elif isinstance(
                        parent.classExpression, ObjectUnionOf
                    ) or isinstance(
                        parent.classExpression, ObjectIntersectionOf
                    ):
                        _set_slot_usage_range(parent.classExpression)
                    else:
                        self.logger.error(
                            "Cannot yet handle anonymous ranges:"
                            + f" {parent.classExpression}"
                        )
                elif isinstance(parent, ObjectSomeValuesFrom):
                    p = self.get_name(parent.objectPropertyExpression)
                    _set_cardinality(p, 1, None)
                    if isinstance(parent.classExpression, Class):
                        cn = self.get_name(parent.classExpression)
                        _set_slot_usage(p, "range", cn)
                    elif isinstance(
                        parent.classExpression, ObjectUnionOf
                    ) or isinstance(
                        parent.classExpression, ObjectIntersectionOf
                    ):
                        _set_slot_usage_range(parent.classExpression)
                    else:
                        self.logger.error(
                            "Cannot yet handle anonymous ranges:"
                            + f" {parent.classExpression}"
                        )
                elif isinstance(parent, DataSomeValuesFrom):
                    if len(parent.dataPropertyExpressions) == 1:
                        p = self.get_name(parent.dataPropertyExpressions[0])
                        _set_cardinality(p, 1, None)
                    else:
                        self.logger.error(
                            "Cannot handle multiple data property"
                            + f" expressions: {parent}"
                        )
                elif isinstance(parent, DataAllValuesFrom):
                    if len(parent.dataPropertyExpressions) == 1:
                        p = self.get_name(parent.dataPropertyExpressions[0])
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
                    p = self.get_name(parent.dataPropertyExpression)
                    lit = parent.literal.v
                    if isinstance(lit, TypedLiteral):
                        lit = lit.literal
                    _set_slot_usage(p, "equals_string", str(lit))
                else:
                    self.logger.error(
                        f"cannot handle anon parent classes for {parent}",
                    )
                # end switch cases

        for c, parents in self.subclassof.items():
            parents = list(parents)
            p = parents.pop()
            self.add_class_info(c, "is_a", p)

            for p in parents:
                self.add_class_info(c, "mixins", p, True)

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

    def process_annotation_assertion(self, a: AnnotationAssertion):
        """Process annotation assertions."""
        p = a.property
        strp = str(p)
        sub = a.subject.v
        val = a.value.v
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
        for cn, usage in self.slot_usage_map.items():
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

    def del_by_name(self, slots, del_list):
        """Delete slots by name."""
        for sn in del_list:
            slots.pop(sn)

    def remove_redundant_slots(self, slots: dict):
        """Remove redundant slots from the schema."""
        self.logger.info("Removing redundant slots...")
        # first trim the slots that are the same as another slot
        del_list = []
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
                    del_list.append(sn_other)
        self.del_by_name(slots, del_list)

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

    def add_slot_to_class(self, slots: dict, assign_result: dict):
        """Add slots to classes.

        Only add a slot to a class if the slot's range is the class or if the
        slot's range is Any. If the slot's range is a union of classes, add the
        slot to the class if the class is one of the classes in the union.
        """
        self.logger.info("Adding slots to classes...")
        for cn, class_info in self.schema["classes"].items():
            if cn in self.schema["slots"] or cn in self.all_properties:
                # skip classes that are slots
                continue
            if "abstract" in class_info and class_info["abstract"]:
                # skip abstract classes, e.g., linkml:Any
                continue

            if "slots" not in class_info:
                class_info["slots"] = []  # initiate the slots list
            for sn, slot_info in slots.items():
                append_this = False
                slot_domain = slot_info.get("domain", "Any")  # default to Any
                if slot_domain == "Any":
                    if "any_of" in slot_info:
                        for x in slot_info[
                            "any_of"
                        ]:  # slot_info["any_of"] is a list of dicts
                            if "domain" in x and cn == x["domain"]:
                                append_this = True
                                break
                    else:
                        # use agentic ai to determine whether to assign the
                        # slot when the domain is Any (unspecified)
                        append_this = assign_result.get((cn, sn), True)
                else:
                    if cn == slot_domain:
                        append_this = True
                if (
                    "slot_usage" in class_info
                    and sn in class_info["slot_usage"]
                ):
                    # force the slot to be added if it is in the slot_usage
                    append_this = True
                if append_this and sn not in class_info["slots"]:
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
        ttl = None
        if filepath.suffix == ".ttl":
            ttl = file  # use ttl for prefixes
            # execute system command
            ofn_filepath = filepath.with_suffix(".ofn")
            if not ofn_filepath.exists():
                self.logger.info(f"Converting {filepath} to .ofn using ROBOT")
                if os.system(
                    f"robot convert -i {str(filepath)} "
                    + f"-output {str(ofn_filepath)}"
                ):
                    raise Exception("Error converting to .ofn")
        elif filepath.suffix != ".ofn":
            raise Exception(f"Invalid file extension: {filepath.suffix}")
        else:
            ofn_filepath = filepath

        doc, ontology = self.read_ofn_file(str(ofn_filepath))

        if name is None:
            name = self.get_name(ontology.iri)
        if ttl is not None:
            self.prefix_dict, self.inv_prefix_dict = self.extract_prefixes_ttl(
                ttl, name
            )
        else:
            self.prefix_dict, self.inv_prefix_dict = self.extract_prefixes(
                doc, name
            )

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

        # add the Any class, for use in ranges
        self.add_class_info("Any", "class_uri", "linkml:Any")
        self.add_class_info("Any", "abstract", True)

        prop_type_list = [
            ObjectProperty,
            DataProperty,
            AnnotationProperty,
        ]

        # iterate over the axioms and process them
        self.logger.info(f"Processing axioms in {file}...")
        for a in ontology.axioms:
            # self.logger.debug(f"Axiom: {a}")

            # process Declaration declarations
            if isinstance(a, Declaration):
                e = a.v
                uri_as_curie = str(e.v)
                if uri_as_curie.startswith(":"):
                    uri_as_curie = f"{name}{uri_as_curie}"
                if isinstance(e, Class) and type(e) not in prop_type_list:
                    cn = self.get_name(e)
                    self.add_class_info(cn, "class_uri", uri_as_curie)
                    self.logger.debug(f"Adding {cn} to classes.")
                if type(e) in [ObjectProperty]:
                    cn = self.get_name(e.v)
                    self.add_slot_info(cn, "slot_uri", uri_as_curie)
                    self.logger.debug(f"Adding {cn} to slots.")
                if type(e) in prop_type_list:
                    cn = self.get_name(e)
                    self.all_properties[cn] = uri_as_curie

            # process the SubClassOf declarations
            if isinstance(a, SubClassOf):
                if isinstance(
                    a.subClassExpression, Class
                ):  # SubClassOf(Class, xxxx)
                    child = self.get_name(a.subClassExpression)

                    if isinstance(
                        a.superClassExpression, ObjectIntersectionOf
                    ) or isinstance(a.superClassExpression, ObjectUnionOf):
                        self.subclassof_list[child] += self.expand(
                            a.superClassExpression
                        )
                    else:
                        self.subclassof_list[child].append(
                            a.superClassExpression
                        )
                else:
                    self.logger.error(
                        f"cannot handle anon child classes for {a}"
                    )

            # process the SubObjectPropertyOf declarations
            # https://github.com/hsolbrig/funowl/issues/19
            if isinstance(a, SubObjectPropertyOf):
                sub = a.subObjectPropertyExpression.v
                if isinstance(sub, ObjectPropertyExpression) and isinstance(
                    sub.v, ObjectProperty
                ):
                    child = self.get_name(sub.v)
                    sup = a.superObjectPropertyExpression.v
                    if isinstance(
                        sup, ObjectPropertyExpression
                    ) and isinstance(sup.v, ObjectProperty):
                        parent = self.get_name(sup.v)
                        self.slot_isamap[child].add(parent)
                    else:
                        self.logger.error(
                            "cannot handle anon object parent"
                            + f" properties for {a}"
                        )
                else:
                    self.logger.error(
                        f"cannot handle anon object child properties for {a}"
                    )

            # process SubDataPropertyOf declarations
            if isinstance(a, SubDataPropertyOf):
                sub = a.subDataPropertyExpression.v
                if isinstance(sub, DataProperty):
                    child = self.get_name(sub)
                    sup = a.superDataPropertyExpression.v
                    if isinstance(sup, DataProperty):
                        parent = self.get_name(sup)
                        self.slot_isamap[child].add(parent)
                    else:
                        self.logger.error(
                            "cannot handle anon data parent properties"
                            + f" for {a}"
                        )
                else:
                    self.logger.error(
                        f"cannot handle anon data child properties for {a}"
                    )

            # process SubAnnotationPropertyOf declarations
            if isinstance(a, SubAnnotationPropertyOf):
                child = self.get_name(a.sub)
                parent = self.get_name(a.super)
                self.slot_isamap[child].add(parent)

            # process the domain and range declarations
            # domains become slot declarations
            if isinstance(a, ObjectPropertyDomain) or isinstance(
                a, DataPropertyDomain
            ):
                if isinstance(a, ObjectPropertyDomain):
                    p = a.objectPropertyExpression.v
                else:
                    p = a.dataPropertyExpression.v
                sn = self.get_name(p)
                dc = a.classExpression
                if isinstance(dc, Class):
                    c = self.get_name(dc)
                    self.add_class_info(c, "slots", sn, True)
                    # logger.error(f'Inferred {c} from domain of {p}')
                if isinstance(dc, ObjectUnionOf):
                    for x in dc.classExpressions:
                        if isinstance(x, Class):
                            c = self.get_name(x)
                            self.add_class_info(c, "slots", sn, True)

            # process ObjectPropertyRange declarations
            if isinstance(a, ObjectPropertyRange):
                p = a.objectPropertyExpression.v
                sn = self.get_name(p)
                rc = a.classExpression
                if isinstance(rc, Class):
                    self.add_slot_info(sn, "range", self.get_name(rc))

            # process DataPropertyRange declarations
            if isinstance(a, DataPropertyRange):
                p = a.dataPropertyExpression.v
                sn = self.get_name(p)
                rc = a.dataRange
                if isinstance(rc, Datatype):
                    self.logger.error("TODO")
                    # self.slot_info(sn, 'range', self.get_name(rc))

            # process AnnotationPropertyRange declarations
            if isinstance(a, AnnotationPropertyRange):
                self.add_slot_info(
                    self.get_name(a.property),
                    "range",
                    self.get_name(a.range),
                )

            if isinstance(a, AnnotationAssertion):
                self.process_annotation_assertion(a)

        # process the parent classes and extract the slot information
        self.process_subclassof()
        self.process_slot()

        slots = deepcopy(self.schema["slots"])
        self.remove_redundant_slots(slots)
        self.remove_irrelevant_slots(slots)
        self.process_slot_usage(slots)
        assign_result = self.query_slot_assignment(slots)
        self.add_slot_to_class(slots, assign_result)
        # self.schema["slots"] = slots
        self.add_identifier(identifier)
        self.rename_forbidden_class_names()
        schema = SchemaDefinition(**self.schema)
        return schema


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
