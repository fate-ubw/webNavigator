from typing import Dict, List, Optional, Set, Tuple
import json
import hashlib

class StatefulTreeNode:
    def __init__(self, accessibility_node: Dict, dom_node: Dict, properties: Dict = None):
        self.accessibility_node = accessibility_node or {}
        self.dom_node = dom_node or {}
        self.node_properties = properties or {}

        def _extract_ax_value(key: str, default=None):
            value = self.accessibility_node.get(key, default)
            if isinstance(value, dict):
                return value.get('value', default)
            return value if value is not None else default

        def _extract_ax_property(prop_name: str, default=None):
            for prop in self.accessibility_node.get('properties', []) or []:
                if prop.get('name') == prop_name:
                    ax_value = prop.get('value')
                    if isinstance(ax_value, dict):
                        return ax_value.get('value', default)
                    return ax_value if ax_value is not None else default
            return default

        def _attributes_to_dict(raw_attrs):
            attrs = {}
            if not raw_attrs:
                return attrs
            # raw_attrs format: [name1, value1, name2, value2, ...]
            for i in range(0, len(raw_attrs), 2):
                name = raw_attrs[i]
                value = raw_attrs[i + 1] if i + 1 < len(raw_attrs) else ''
                attrs[name] = value
            return attrs

        def _computed_style_to_dict(raw_styles):
            styles = {}
            if not raw_styles:
                return styles
            for style in raw_styles:
                name = style.get('name')
                value = style.get('value')
                if name:
                    styles[name] = value
            return styles

        self.attributes = _attributes_to_dict(self.dom_node.get('attributes'))
        self.computed_styles = _computed_style_to_dict(self.node_properties.get('computed_styles'))

        # Structural information.
        local_name = self.dom_node.get('localName') or self.dom_node.get('nodeName') or ''
        self.tag_name: str = local_name.lower() or None
        self.role: str = _extract_ax_value('role')
        self.accessible_name: str = _extract_ax_value('name')

        # State information.
        # hidden = bool(_extract_ax_property('hidden', False))
        # ignored = self.accessibility_node.get('ignored', False)
        # display_none = self.computed_styles.get('display') == 'none'
        # visibility_hidden = self.computed_styles.get('visibility') == 'hidden'
        # opacity_zero = self.computed_styles.get('opacity') == '0'
        # self.is_visible: bool = not any([hidden, ignored, display_none, visibility_hidden, opacity_zero])

        disabled = bool(_extract_ax_property('disabled', False))
        readonly = bool(_extract_ax_property('readonly', False))
        enabled = _extract_ax_property('enabled')
        self.is_enabled: bool = enabled if enabled is not None else not disabled
        self.is_readonly: bool = readonly
        self.is_focusable: bool = bool(_extract_ax_property('focusable', False))

        # Content information.
        self.text_content: str = self.dom_node.get('nodeValue') or _extract_ax_property('value') or ''
        self.description: str = _extract_ax_value('description') or ''
        self.placeholder: str = self.attributes.get('placeholder')
        self.href: str = self.attributes.get('href')
        form_value = self.attributes.get('value')
        if form_value is None:
            form_value = _extract_ax_property('value')
        self.value: str = form_value
        self.semantic_reserved: str = ""

        # Geometry information.
        self.box_model = self.node_properties.get('box_model') or {}

        # Tree structure.
        self.id: str = accessibility_node.get('nodeId')
        self.parent_id: str = accessibility_node.get('parentId')
        self.child_ids: List[str] = accessibility_node.get('childIds')
        self.xpath: str = accessibility_node.get('xpath')

        # Hash cache.
        self._content_hash: str = hashlib.md5(str(self.text_content).encode('utf-8') + str(self.placeholder).encode('utf-8') + str(self.href).encode('utf-8') + str(self.value).encode('utf-8') + str(self.description).encode('utf-8')).hexdigest()
        self._structure_hash: str = hashlib.md5(str(self.xpath).encode('utf-8')).hexdigest()

    def get_content_hash(self) -> str:
        return self._content_hash

    def get_structure_hash(self) -> str:
        return self._structure_hash

    def __hash__(self) -> int:
        return hash(self._content_hash, self._structure_hash)

class StatefulTree:
    def __init__(self, accessibility_dom_properties_compound: List[Tuple[Dict, Dict, Dict]]):
        self.nodes: Dict[str, StatefulTreeNode] = {}
        for accessibility_node, dom_node, properties in accessibility_dom_properties_compound:
            node = StatefulTreeNode(accessibility_node, dom_node, properties)
            self.nodes[node.id] = node
        self._structure_hash_set: Set[str] = set(node.get_structure_hash() for node in self.nodes.values())
        self._content_hash_set: Set[str] = set(node.get_content_hash() for node in self.nodes.values())

    def get_all_nodes_by_role(self, role: str) -> Dict[str, StatefulTreeNode]:
        nodes = {}
        for node in self.nodes.values():
            if node.role == role:
                nodes[node.id] = node
        return nodes

    def get_all_nodes_by_tag_name(self, tag_name: str) -> Dict[str, StatefulTreeNode]:
        nodes = {}
        for node in self.nodes.values():
            if node.tag_name == tag_name:
                nodes[node.id] = node
        return nodes

    def get_node_by_structure_hash(self, structure_hash: str) -> Optional[StatefulTreeNode]:
        for node in self.nodes.values():
            if node.get_structure_hash() == structure_hash:
                return node
        return None

    def get_structure_hash_set(self) -> Set[str]:
        return self._structure_hash_set

    def get_content_hash_set(self) -> Set[str]:
        return self._content_hash_set
