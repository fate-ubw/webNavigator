import hashlib
from time import sleep
from typing import Dict, List
from playwright.sync_api import Locator, Page, TimeoutError

from .web_node import WebNode

class WebActionObject:
    def __init__(
        self,
        action_object_type: str,
        action_object_identifier: str,
    ):
        self.action_object_type = action_object_type
        self.action_object_identifier = action_object_identifier
        self._unique_id = hashlib.md5(str(self.action_object_type).encode('utf-8') + str(self.action_object_identifier).encode('utf-8')).hexdigest()

    def get_unique_id(self) -> str:
        return self._unique_id

class WebActionEntry:
    def __init__(
        self,
        action_type: str,
        action_object: WebActionObject,
        action_parameters: dict,
    ):
        self.action_type = action_type
        self.action_object = action_object
        self.action_parameters = action_parameters
        self._unique_id = hashlib.md5(str(self.action_type).encode('utf-8') + str(self.action_object.get_unique_id()+ str(self.action_parameters)).encode('utf-8')).hexdigest()

    def __perform_page_element_action(self, page_element: Locator):
        if self.action_type == 'click':
            page_element.click()
        elif self.action_type == 'input':
            page_element.fill(self.action_parameters['value'])
        elif self.action_type == 'select':
            page_element.select_option(self.action_parameters['value'])

    def do_action(self, page: Page, current_node: WebNode):
        if self.action_object.action_object_type == 'StatefulTreeNode':
            if self.action_object.action_object_identifier.startswith('structure_hash:'):
                structure_hash = self.action_object.action_object_identifier.split(':')[1]
                axtree_element = current_node.stateful_tree_instance.get_node_by_structure_hash(structure_hash)
                if axtree_element is None:
                    raise ValueError(f"Node with structure hash {structure_hash} not found")
                page_element = page.locator("xpath=" + axtree_element.xpath)
                self.__perform_page_element_action(page_element)
        elif self.action_object.action_object_type == 'WebMetadata':
            if self.action_type == 'goto':
                try:
                    page.goto(self.action_parameters['url'], wait_until='networkidle')
                except TimeoutError as e:
                    page.evaluate('() => window.stop()')

    def get_unique_id(self) -> str:
        return self._unique_id

class WebAction:
    def __init__(
        self,
        action_entries: List[WebActionEntry],
        ):
        self.action_entries = action_entries
        self._unique_id = hashlib.md5(str([entry.get_unique_id() for entry in self.action_entries]).encode('utf-8')).hexdigest()

    def get_unique_id(self) -> str:
        return self._unique_id

    def perform_action(self, page: Page, current_node: WebNode):
        for action_entry in self.action_entries:
            page.wait_for_timeout(2000)
            action_entry.do_action(page, current_node)

def new_web_action(
    action_entries: List[Dict],
):
    entries = []
    for action_entry in action_entries:
        entries.append(WebActionEntry(action_entry['action_type'], WebActionObject(action_entry['action_object_type'], action_entry['action_object_identifier']), action_entry['action_parameters']))
    return WebAction(entries)