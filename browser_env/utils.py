from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union
from urllib.parse import urljoin, urlparse

import numpy as np
import numpy.typing as npt
from PIL import Image
from playwright.sync_api import CDPSession


@dataclass
class DetachedPage:
    url: str
    content: str  # html


def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]

def get_enhanced_ax_tree(client: CDPSession):
    """
    Get the full accessibility tree (AXTree) with injected URL attributes.

    Approach:
    1. Call Accessibility.getFullAXTree and DOM.getDocument.
    2. Convert the flat DOM attributes list to a dict and build a
       {backendDOMNodeId -> attributes} index.
    3. Traverse the AXTree, match DOM attributes via backendDOMNodeId,
       build absolute URLs, and inject the 'url' field.
    
    Args:
        client: Playwright CDPSession object.
        
    Returns:
        List[dict]: AXNode list with injected 'url' fields.
    """
    
    # 1. Send CDP commands and include iframe/shadow DOM nodes.
    ax_tree = client.send("Accessibility.getFullAXTree")
    root_dom = client.send("DOM.getDocument", {"depth": -1, "pierce": True})

    root_node = root_dom.get('root', {})
    base_url = root_node.get('baseURL') or root_node.get('documentURL') or ""
    base_netloc = urlparse(base_url).netloc if base_url else ""
    if not base_url:
        print("Warning: Could not retrieve Base URL from DOM.")
    
    # 2. Build DOM lookup index: { backendId: {href: "...", src: "..."} }
    dom_lookup = {}

    def _process_dom_node(node):
        # Extract backendDOMNodeId.
        node_id = node.get('backendNodeId')
        
        # Convert CDP flat attributes ['href', '/foo', 'id', 'bar'] to a dict.
        attrs_list = node.get('attributes', [])
        if node_id and attrs_list:
            attrs_dict = {}
            # Iterate in steps of 2.
            for i in range(0, len(attrs_list), 2):
                key = attrs_list[i]
                val = attrs_list[i+1]
                attrs_dict[key] = val
            dom_lookup[node_id] = attrs_dict

        # Recursively process child nodes.
        for child in node.get('children', []):
            _process_dom_node(child)
            
        # Recursively process iframe content documents.
        if 'contentDocument' in node:
            _process_dom_node(node['contentDocument'])

    # Start processing the DOM tree.
    if 'root' in root_dom:
        _process_dom_node(root_dom['root'])

    # 3. Traverse AXTree and inject URLs.
    nodes = ax_tree.get('nodes', [])
    
    for ax_node in nodes:
        # Get the DOM ID mapped to this AX node.
        dom_id = ax_node.get('backendDOMNodeId')
        
        if dom_id and dom_id in dom_lookup:
            dom_attrs = dom_lookup[dom_id]
            
            # Try href (usually links) or src (usually images).
            raw_url = dom_attrs.get('href') or dom_attrs.get('src')
            
            if raw_url:
                # Filter empty values and javascript: pseudo-protocol URLs.
                if raw_url.strip() and not raw_url.lower().startswith('javascript:'):
                    # Build the absolute URL.
                    full_url = urljoin(base_url, raw_url)
                    # Skip cross-origin links; only inject same-origin URLs.
                    parsed_full = urlparse(full_url)
                    if base_netloc and parsed_full.netloc and parsed_full.netloc != base_netloc:
                        continue
                    # Inject URL directly at AXNode top level for downstream use.
                    ax_node['url'] = full_url
                    # Uncomment the next line to also keep the original relative path.
                    # ax_node['rawHref'] = raw_url
    return nodes