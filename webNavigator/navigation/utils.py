import urllib.parse
from typing import Any, List, Dict, Optional, Tuple, Union
import os
from playwright.sync_api import Page, TimeoutError
import pickle
from .element_matcher import DOMElementMatcher
from .constants import EXPLORE_RESULTS_DIR, SCREENSHOT_DIR
import requests
import json
import base64
from pathlib import Path
from scipy.spatial.distance import cosine
import heapq
import numpy as np

def get_hostname(url: str) -> Optional[str]:
    """Extract the hostname component from a URL string."""
    if not url:
        return None

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        parsed = urllib.parse.urlparse(f"//{url}", scheme="http")

    return parsed.netloc

def classify_urls(urls: List[str], current_domain: str = None, current_url: str = None) -> Dict[str, List[str]]:
    """
    Classify URLs into internal, external, anchor, and invalid categories.
    
    Args:
        urls: URL list to classify.
        current_domain: Current page domain (optional, for better internal/external judgment).
        current_url: Current page full URL (optional, for resolving relative paths).
    
    Returns:
        Classification result dict with keys: 'internal', 'external', 'anchor', 'invalid'.
    """
    
    result = {
        'internal': [],
        'external': [],
        'anchor': [],
        'invalid': []
    }
    
    # If current_url is provided, derive domain from it.
    if current_url and not current_domain:
        try:
            parsed_current = urllib.parse.urlparse(current_url)
            current_domain = parsed_current.netloc
        except:
            pass
    
    for url in urls:
        url = url.strip()
        
        # Empty string is invalid.
        if not url or url == '':
            result['invalid'].append(url)
            continue
            
        # Anchor links that start with '#', including '#' itself.
        if url.startswith('#'):
            result['anchor'].append(url)
            continue
            
        # Relative path (starts with '/' but not '//').
        if url.startswith('/') and not url.startswith('//'):
            if current_domain:
                result['internal'].append(url)
            else:
                # If current domain is unknown, treat relative paths as internal.
                result['internal'].append(url)
            continue
            
        try:
            parsed_url = urllib.parse.urlparse(url)
            
            if not parsed_url.scheme:
                if parsed_url.path:
                    result['internal'].append(url)
                else:
                    result['invalid'].append(url)
                continue
                
            if parsed_url.netloc:
                if current_domain:
                    # Compare host to decide internal vs external.
                    if parsed_url.netloc.lower() == current_domain.lower():
                        result['internal'].append(url)
                    else:
                        result['external'].append(url)
                else:
                    # If current domain is unknown, treat complete URLs as external.
                    result['external'].append(url)
            else:
                result['invalid'].append(url)
                
        except Exception as e:
            result['invalid'].append(url)
            
    return result

def classify_url_simple(url: str, current_domain: str = None) -> str:
    """
    Simplified helper: return a single URL classification result.
    
    Args:
        url: URL to classify.
        current_domain: Current page domain (optional).
    
    Returns:
        URL type: 'internal', 'external', 'anchor', 'invalid'.
    """
    result = classify_urls([url], current_domain)
    
    for category, urls in result.items():
        if urls:
            return category
    
    return 'invalid'

def get_page_screenshot(page: Page, filename: str, new_dir_name: str = "default"):
    try:
        page.wait_for_load_state('load')
    except TimeoutError as e:
        page.evaluate('() => window.stop()')
    path = os.path.join(EXPLORE_RESULTS_DIR, new_dir_name, SCREENSHOT_DIR)
    os.makedirs(path, exist_ok=True)
    if not filename.endswith('.png'):
        filename += '.png'
    page.wait_for_timeout(2000)
    page.screenshot(path=os.path.join(path, filename), full_page=True)
    return os.path.join(path, filename)


def save_graph(graph, new_dir_name: str):
    path = os.path.join(EXPLORE_RESULTS_DIR, new_dir_name)
    os.makedirs(path, exist_ok=True)
    file_name = 'web_graph.pkl'
    file_path = os.path.join(path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f) 

def load_graph(path: str):
    file_name = 'web_graph.pkl'
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

BLOCKLIST_CONFIG = json.load(open("/".join([os.path.dirname(__file__), 'blocklist.json']), 'r', encoding='utf-8'))
DOM_ELEMENT_MATCHER = DOMElementMatcher()
DOM_ELEMENT_MATCHER.load_config(BLOCKLIST_CONFIG)

def is_element_blocked(element: Dict[str, Any], domain: str = None) -> bool:
    results = DOM_ELEMENT_MATCHER.match_candidates([element], domain)
    # if len(results["product_actions"]) == 0:
    #     return True
    # print(element)
    is_in = False
    for result in results.values():
        if len(result) > 0:
            is_in = True
            break
    return is_in