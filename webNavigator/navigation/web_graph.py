from typing import Dict, List, Tuple

from playwright.sync_api import Page, TimeoutError
from .web_action import WebAction
from .web_node import WebNode


class WebGraph:
    def __init__(self):
        self.nodes: Dict[str, WebNode] = {}
        self.actions: Dict[str, WebAction] = {}
        self.edges: List[Tuple[str, str, str]] = []

    def add_node(self, node: WebNode):
        if node.get_unique_id() in self.nodes:
            return
        else:
            self.nodes[node.get_unique_id()] = node

    def add_edge(self, from_node: WebNode, to_node: WebNode, action: WebAction) -> bool:
        if(from_node.get_unique_id() not in self.nodes):
            raise ValueError(f"Node {from_node.get_unique_id()} not found in graph. This is not possible.")
        if(to_node.get_unique_id() in self.nodes):
            return False
        self.add_node(to_node)
        self.__add_action(action)
        self.edges.append((from_node.get_unique_id(), to_node.get_unique_id(), action.get_unique_id()))
        return True

    def __add_action(self, action: WebAction):
        self.actions[action.get_unique_id()] = action

    def find_edge_by_to_node(self, to_node: WebNode) -> Tuple[str, str, str] | None:
        for edge in self.edges:
            if edge[1] == to_node.get_unique_id():
                return edge
        return None

    def navigate_to_node(self, page: Page, node: WebNode):
        if node.can_arrival_directly:
            try:
                page.goto(node.web_metadata_instance.url, wait_until='load')
            except TimeoutError as e:
                page.evaluate('() => window.stop()')
            return True
        else:
            path = []
            current_edge = self.find_edge_by_to_node(node)
            while current_edge:
                path.append(current_edge)
                if(self.get_node(current_edge[0]).can_arrival_directly):
                    break
                current_edge = self.find_edge_by_to_node(self.get_node(current_edge[0]))
            path.reverse()
            try:
                page.goto(self.get_node(path[0][0]).web_metadata_instance.url, wait_until='load')
            except TimeoutError as e:
                page.evaluate('() => window.stop()')
            for edge in path:
                action = self.get_action(edge[2])
                try:
                    action.perform_action(page, self.get_node(edge[0]))
                except:
                    try:
                        response = page.goto(node.web_metadata_instance.url, wait_until="commit")
                        if response is None:
                            return False
                        if response.status >= 400:
                            return False
                        page.wait_for_load_state('load')
                    except TimeoutError as e:
                        page.evaluate('() => window.stop()')
            return True

    def get_node(self, unique_id: str) -> WebNode:
        return self.nodes[unique_id]

    def get_action(self, unique_id: str) -> WebAction:
        return self.actions[unique_id]

    def __repr__(self) -> str:
        return f"WebGraph(nodes={len(self.nodes)}, actions={len(self.actions)}, edges={len(self.edges)})"