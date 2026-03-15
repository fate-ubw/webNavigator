"""
DomainNavigator: Encapsulates all navigation resources and capabilities for a single domain
Integrates retrieval + webgraph + future additional functionality
"""

import os
import pickle
import argparse
import urllib.parse
from typing import Any

from .retrieval import JinaRetriever
# Import navigation module, this sets sys.modules alias for pickle to load correctly
from . import navigation
from .navigation import WebGraph
from .utils import load_images, images_to_base64


class DomainNavigator:
    """
    Encapsulates all navigation resources and capabilities for a single domain
    
    Attributes:
        name: domain name (e.g., "shopping_admin", "map", "shopping", "gitlab", "reddit")
        base_path: data directory (e.g., "webNavigator/webNodes/run_id-xxx-admin-124")
        retriever: JinaRetriever instance, used for retrieval
        webgraph: WebGraph instance, used for navigation (TODO: to be completed)
        screenshots: list of screenshot images in base64 format for LLM input
    """
    
    def __init__(self, name: str, base_path: str, args: argparse.Namespace):
        """
        Args:
            name: domain name
            base_path: data directory path, containing image_embedding.pkl and web_graph.pkl
            args: configuration parameters, including topk, return_multivector, etc.
        """
        self.name = name
        self.base_path = base_path
        self.args = args
        
        # Resources
        self.retriever: JinaRetriever | None = None
        self.webgraph: WebGraph | None = None
        self.screenshots: list[dict[str, str]] = []  # base64 encoded images for LLM input
        
        # Load all resources
        self._load_all()

    
    def _load_all(self) -> None:
        """Load all resources"""
        self._load_retriever()
        self._load_webgraph()
        self._load_screenshots()
        self.post_process()
    
    def post_process(self) -> None:
        """Post-process the webgraph: replace URLs with correct base URL and port from environment"""
        if self.webgraph is None:
            return
        
        # Resolve environment variable name from self.name, e.g. gitlab -> GITLAB.
        env_var_name = self.name.upper()
        domain_url = os.environ.get(env_var_name)
        
        if not domain_url:
            print(f"[DomainNavigator:{self.name}] Warning: {env_var_name} environment variable not set, skipping URL replacement")
            return
        # Parse domain URL, e.g. "http://210.28.133.13:22506" or ".../admin".
        parsed_domain = urllib.parse.urlparse(domain_url)
        new_scheme = parsed_domain.scheme  # http
        new_netloc = parsed_domain.netloc  # 210.28.133.13:22506
        domain_base_path = parsed_domain.path  # Might be "" or "/admin".
        
        replaced_count = 0
        for node_id, node in self.webgraph.nodes.items():
            metadata = node.web_metadata_instance
            old_url = metadata.url
            
            # Parse original URL.
            parsed_old = urllib.parse.urlparse(old_url)
            old_path = parsed_old.path  # e.g. "/admin/dashboard" or "/-/profile/gpg_keys"
            
            # Build the new path.
            # If domain URL contains a base path (e.g. /admin), handle it carefully.
            if domain_base_path and old_path.startswith(domain_base_path):
                # If original path already includes domain base path, keep it.
                new_path = old_path
            elif domain_base_path:
                # If domain has a base path but original path does not, prepend it.
                new_path = domain_base_path.rstrip('/') + '/' + old_path.lstrip('/')
            else:
                # If no domain base path is set, keep original path.
                new_path = old_path
            
            # Build the new URL.
            new_url = urllib.parse.urlunparse((
                new_scheme,             # http
                new_netloc,             # 210.28.133.13:22506
                new_path,               # /-/profile/gpg_keys
                parsed_old.params,
                parsed_old.query,
                parsed_old.fragment
            ))
            
            # Update metadata.
            metadata.url = new_url
            metadata.host = new_netloc
            replaced_count += 1
            # print(f"[DomainNavigator:{self.name}] Replaced URL for node {node_id} from {old_url} to {new_url}")
        
        print(f"[DomainNavigator:{self.name}] Replaced URLs for {replaced_count} nodes (using {env_var_name}={domain_url})")

    
    def _load_retriever(self) -> None:
        """Load JinaRetriever"""
        embedding_path = os.path.join(self.base_path, "image_embedding.pkl")
        
        if os.path.exists(embedding_path):
            # Set parameters needed for JinaRetriever
            self.args.persistence_file = embedding_path
            print(f"[DomainNavigator:{self.name}] Loading retriever from {embedding_path}")
            self.retriever = JinaRetriever(self.args)
        else:
            print(f"[DomainNavigator:{self.name}] Warning: embedding file not found: {embedding_path}")
            self.retriever = None
    
    def _load_webgraph(self) -> None:
        """Load WebGraph"""
        graph_path = os.path.join(self.base_path, "web_graph.pkl")
        
        if os.path.exists(graph_path):
            print(f"[DomainNavigator:{self.name}] Loading webgraph from {graph_path}")
            with open(graph_path, 'rb') as f:
                self.webgraph = pickle.load(f)
        else:
            print(f"[DomainNavigator:{self.name}] Warning: webgraph file not found: {graph_path}")
            self.webgraph = None
    
    def _load_screenshots(self) -> None:
        """Load screenshot images and convert to base64 for LLM input"""
        screenshots_path = os.path.join(self.base_path, "screenshots")
        
        if os.path.exists(screenshots_path) and os.path.isdir(screenshots_path):
            print(f"[DomainNavigator:{self.name}] Loading screenshots from {screenshots_path}")
            raw_images = load_images(screenshots_path)
            self.screenshots = images_to_base64(raw_images)
            print(f"[DomainNavigator:{self.name}] Loaded {len(self.screenshots)} screenshots")
        else:
            print(f"[DomainNavigator:{self.name}] Warning: screenshots folder not found: {screenshots_path}")
            self.screenshots = []
    



    def __repr__(self) -> str:
        screenshots_status = f"✓({len(self.screenshots)})" if self.screenshots else "✗"
        return f"DomainNavigator(name='{self.name}', retriever={f'✓{self.retriever}' if self.retriever else '✗'}, webgraph={f'✓{self.webgraph}' if self.webgraph else '✗'}, screenshots={screenshots_status})"

