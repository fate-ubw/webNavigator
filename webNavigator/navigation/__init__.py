"""
Navigation module for WebNavigator
Contains classes related to graph topology, such as WebGraph, WebNode, and WebAction.

Note: to correctly load previously saved `web_graph.pkl` files with pickle,
we set `sys.modules` aliases so pickle can resolve the original module names.
"""

import sys

# Import modules first.
from . import web_graph
from . import web_node
from . import web_action
from . import web_metadata
from . import stateful_tree

# Set sys.modules aliases so pickle can resolve historical module names.
# Older pkl files were saved with module names like "web_graph".
sys.modules['web_graph'] = web_graph
sys.modules['web_node'] = web_node
sys.modules['web_action'] = web_action
sys.modules['web_metadata'] = web_metadata
sys.modules['stateful_tree'] = stateful_tree

# Export primary classes.
from .web_graph import WebGraph
from .web_node import WebNode, new_web_node_snapshot
from .web_action import WebAction, WebActionEntry, WebActionObject
from .web_metadata import WebMetadata
from .stateful_tree import StatefulTree, StatefulTreeNode

__all__ = [
    "WebGraph",
    "WebNode",
    "WebAction",
    "WebActionEntry",
    "WebActionObject",
    "WebMetadata",
    "StatefulTree",
    "StatefulTreeNode",
    "new_web_node_snapshot",
]

