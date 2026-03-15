import hashlib
from typing import List, Tuple

from playwright.sync_api import Page, TimeoutError
from .browser_backend_chrome import BrowserBackendChrome
from .stateful_tree import StatefulTree
from .web_metadata import WebMetadata

class WebNode:
    def __init__(
        self,
        stateful_tree_instance: StatefulTree,
        web_metadata_instance: WebMetadata,
        ):
        self.stateful_tree_instance = stateful_tree_instance
        self.web_metadata_instance = web_metadata_instance
        self.can_arrival_directly = False
        self.arrival_necessary_params = {}
        self._unique_id = self.__make_unique_id()
        self.screen_path = None

    def compare_to_structure(self, other: 'WebNode') -> bool:
        return self.stateful_tree_instance.get_structure_hash_set() == other.stateful_tree_instance.get_structure_hash_set()
    
    def compare_to_content(self, other: 'WebNode') -> bool:
        return self.stateful_tree_instance.get_content_hash_set() == other.stateful_tree_instance.get_content_hash_set()
    
    def compare_to_metadata(self, other: 'WebNode') -> bool:
        return self.web_metadata_instance.get_metadata_hash() == other.web_metadata_instance.get_metadata_hash()
    
    def compare_to(self, other: 'WebNode') -> bool:
        return self._unique_id == other._unique_id

    def get_structure_difference(self, other: 'WebNode') -> Tuple[List[str], List[str]]:
        return (self.stateful_tree_instance.get_structure_hash_set() - other.stateful_tree_instance.get_structure_hash_set(), other.stateful_tree_instance.get_structure_hash_set() - self.stateful_tree_instance.get_structure_hash_set())

    def __make_unique_id(self) -> str:
        structure_hash = self.stateful_tree_instance.get_structure_hash_set()
        content_hash = self.stateful_tree_instance.get_content_hash_set()
        content_hash = ""
        metadata_hash = self.web_metadata_instance.get_metadata_hash()
        return hashlib.md5(str(structure_hash).encode('utf-8') + str(content_hash).encode('utf-8') + str(metadata_hash).encode('utf-8')).hexdigest()

    def get_unique_id(self) -> str:
        return self._unique_id

def new_web_node_snapshot(page: Page):
    try:
        page.wait_for_load_state('load') 
    except TimeoutError as e:
        page.evaluate('() => window.stop()')
    backend = BrowserBackendChrome(page)
    backend.initialize_cdp()
    compound_tree = backend.get_compound()
    stateful_tree0 = StatefulTree(compound_tree)
    backend.close()
    url = page.url
    return WebNode(stateful_tree0, WebMetadata(url))

def can_node_arrive_directly(page: Page, node: WebNode, patch: dict = None) -> bool:
    try:
        page.goto(node.web_metadata_instance.url, wait_until='load')
    except TimeoutError as e:
        page.evaluate('() => window.stop()')
    if patch and patch['wait_time']:
        page.wait_for_timeout(patch['wait_time'])
    node2 = new_web_node_snapshot(page)
    return node.compare_to(node2)