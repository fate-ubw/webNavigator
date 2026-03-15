from typing import Any, Dict, List, Optional
from playwright.sync_api import sync_playwright, Page
from tqdm import tqdm

class BrowserBackendChrome:
    def __init__(self, page: Page):
        self.page = page
        self.cdp = None

    def _extract_ax_value(self, accessibility_node: Dict, key: str, default=None):
        value = accessibility_node.get(key, default)
        if isinstance(value, dict):
            return value.get('value', default)
        return value if value is not None else default
        
    def initialize_cdp(self):
        """Initialize CDP connection."""
        self.cdp = self.page.context.new_cdp_session(self.page)
        
        # Enable required CDP domains.
        self.cdp.send('Accessibility.enable')
        self.cdp.send('Runtime.enable')
        self.cdp.send('DOM.enable')
        self.cdp.send('CSS.enable')
        self.cdp.send('DOM.getDocument', {'depth': 0})
        
    def get_xpath_via_page_evaluation(self, backend_dom_node_id: int):

        cdp = self.cdp

        # Push backend node to frontend.
        push_result = cdp.send("DOM.pushNodesByBackendIdsToFrontend", {
            "backendNodeIds": [backend_dom_node_id]
        })
        node_id = push_result["nodeIds"][0]
        if node_id == 0:
            return None
        # Resolve object reference.
        resolve_result = cdp.send("DOM.resolveNode", {
            "nodeId": node_id
        })
        object_id = resolve_result["object"]["objectId"]
        # Execute XPath computation in page context.
        xpath_result = cdp.send("Runtime.callFunctionOn", {
            "objectId": object_id,
            "functionDeclaration": """
                function() {
                    const element = this;
                    
                    function getXPath(el) {
                        if (el.id) {
                            return `//*[@id="${el.id}"]`;
                        }
                        
                        if (el === document.body) {
                            return '/html/body';
                        }
                        
                        if (el === document.documentElement) {
                            return '/html';
                        }
                        
                        let path = '';
                        let current = el;
                        
                        while (current && current.nodeType === Node.ELEMENT_NODE) {
                            let tagName = current.tagName.toLowerCase();
                            let index = 1;
                            
                            // Walk previous siblings to compute index.
                            let sibling = current.previousElementSibling;
                            while (sibling) {
                                if (sibling.tagName.toLowerCase() === tagName) {
                                    index++;
                                }
                                sibling = sibling.previousElementSibling;
                            }
                            
                            // Check whether later siblings share the same tag.
                            let hasNextSibling = false;
                            let nextSibling = current.nextElementSibling;
                            while (nextSibling) {
                                if (nextSibling.tagName.toLowerCase() === tagName) {
                                    hasNextSibling = true;
                                    break;
                                }
                                nextSibling = nextSibling.nextElementSibling;
                            }
                            
                            let pathSegment = tagName;
                            if (index > 1 || hasNextSibling) {
                                pathSegment += `[${index}]`;
                            }
                            
                            path = '/' + pathSegment + path;
                            current = current.parentElement;
                        }
                        
                        return path || '/';
                    }
                    
                    return getXPath(element);
                }
            """,
            "returnByValue": True
        })
        return xpath_result["result"]["value"]

    def get_accessibility_tree(self, skipped_roles: List[str] = ["generic"]) -> Dict[str, Any]:
        """Get full accessibility tree."""
        if not self.cdp:
            self.initialize_cdp()
            
        # Get root nodes.
        result = self.cdp.send('Accessibility.getFullAXTree')
        nodes = result.get('nodes', [])
        nodes = [node for node in nodes if node['ignored'] == False and self._extract_ax_value(node, 'role') not in skipped_roles]
        for node in tqdm(nodes):
            try:
                node['xpath'] = self.get_xpath_via_page_evaluation(node['backendDOMNodeId'])
                if node['xpath'] == "/":
                    node['xpath'] = None
            except Exception as e:
                pass
        nodes = [node for node in nodes if node.get('xpath') is not None]
        return nodes
    def get_dom_node_from_backendDOMNodedId_id(self, backend_dom_node_id: str) -> (str, Dict):
        if not self.cdp:
            self.initialize_cdp()
        
        dom_result = self.cdp.send('DOM.describeNode', {
            'backendNodeId': backend_dom_node_id
        })
        
        return dom_result.get('node')
                    
    
    def get_element_properties(self, backend_node_id: int) -> Dict[str, Any]:
        """Get detailed DOM element properties."""
        if not self.cdp:
            self.initialize_cdp()
            
        try:
            frontend_nodes = self.cdp.send('DOM.pushNodesByBackendIdsToFrontend', {
                'backendNodeIds': [backend_node_id]
            })
            node_id = frontend_nodes['nodeIds'][0]

            styles = self.cdp.send('CSS.getComputedStyleForNode', {
                'nodeId': node_id
            })
            attributes = self.cdp.send('DOM.getAttributes', {
                'nodeId': node_id
            })
            box_model = self.cdp.send('DOM.getBoxModel', {
                'nodeId': node_id
            })
            
            return {
                # 'computed_styles': styles.get('computedStyle', []),
                # 'attributes': attributes.get('attributes', []),
                'box_model': box_model.get('model', {})
            }
            
        except Exception as e:
            # print(f"Error getting element properties: {e}")
            return {}

    def get_compound(self, skipped_roles: List[str] = ["generic"]):
        accessibility_tree = self.get_accessibility_tree(skipped_roles)
        compound_tree = []
        for node in tqdm(accessibility_tree):
            try:
                dom_node = self.get_dom_node_from_backendDOMNodedId_id(node['backendDOMNodeId'])
                properties = {}
                # properties = backend.get_element_properties(dom_node['backendNodeId'])
                compound_tree.append((node, dom_node, properties))
            except:
                pass
        # result0 = self.detect_dynamic_lists(self.page)
        # print(result0)
        return compound_tree

    def detect_dynamic_lists(self, page: Page, min_items: int = 3) -> List[Dict[str, Any]]:
        """
        Detect dynamic list-like regions on the page.
        
        Returns:
            List[Dict]: List of dicts with fields:
            - container_xpath: Full XPath of the container
            - child_tag_class: Child tag+class signature
            - child_internal_structure: Internal relative XPath list of child nodes
            - item_count: Number of child items
        """
        
        js_code = """
        () => {
            const MIN_ITEMS = """+str(min_items)+""";
            
            // Helper: get full XPath of an element.
            function getFullXPath(element) {
                if (!element || element === document.documentElement) {
                    return '/html';
                }
                
                if (element === document.body) {
                    return '/html/body';
                }
                
                const parts = [];
                let current = element;
                
                while (current && current !== document.documentElement) {
                    let tagName = current.tagName.toLowerCase();
                    
                    // Compute index among siblings with the same tag.
                    const siblings = Array.from(current.parentNode.children).filter(
                        sibling => sibling.tagName.toLowerCase() === tagName
                    );
                    
                    if (siblings.length > 1) {
                        const index = siblings.indexOf(current) + 1;
                        tagName += `[${index}]`;
                    }
                    
                    parts.unshift(tagName);
                    current = current.parentElement;
                }
                
                return '/' + parts.join('/');
            }
            
            // Helper: get relative XPath (relative to ancestor).
            function getRelativeXPath(element, ancestor) {
                if (element === ancestor) {
                    return '.';
                }
                
                const parts = [];
                let current = element;
                
                while (current && current !== ancestor) {
                    let tagName = current.tagName.toLowerCase();
                    
                    // Compute index among siblings with the same tag.
                    const siblings = Array.from(current.parentNode.children).filter(
                        sibling => sibling.tagName.toLowerCase() === tagName
                    );
                    
                    if (siblings.length > 1) {
                        const index = siblings.indexOf(current) + 1;
                        tagName += `[${index}]`;
                    }
                    
                    parts.unshift(tagName);
                    current = current.parentElement;
                }
                
                return parts.join('/');
            }
            
            // Helper: build full attribute signature of an element.
            function getElementAttributeSignature(element) {
                const tag = element.tagName.toLowerCase();
                const attributes = {};
                
                // Collect all attributes.
                Array.from(element.attributes).forEach(attr => {
                    const name = attr.name.toLowerCase();
                    let value = attr.value;
                    
                    // Normalize specific attributes.
                    if (name === 'class') {
                        // Sort classes for stable comparison.
                        value = value.trim().split(/\\s+/).filter(cls => cls).sort().join(' ');
                    } else if (name === 'style') {
                        // Ignore inline-style details, keep only presence marker.
                        value = value ? '[has-style]' : '';
                    } else if (name.startsWith('data-') && /^data-.*-[a-f0-9]{6,}$/i.test(name)) {
                        // Ignore dynamically generated data-* attributes.
                        value = '[dynamic-data]';
                    }
                    
                    attributes[name] = value;
                });
                
                // Build attribute signature string.
                const attrKeys = Object.keys(attributes).sort();
                const attrStrings = attrKeys.map(key => `${key}="${attributes[key]}"`);
                
                return `<${tag} ${attrStrings.join(' ')}>`;
            }
            
            // Helper: compare whether two signatures are structurally similar.
            function areAttributeSignaturesSimilar(sig1, sig2) {
                // First check exact equality.
                if (sig1 === sig2) {
                    return true;
                }
                
                // Parse signatures into tag + attribute dict.
                const parseSignature = (sig) => {
                    const match = sig.match(/^<(\\w+)\\s*(.*)>$/);
                    if (!match) return null;
                    
                    const tag = match[1];
                    const attrString = match[2];
                    const attributes = {};
                    
                    // Simple attribute parsing.
                    const attrMatches = attrString.matchAll(/(\\w+(?:-\\w+)*)="([^"]*)"/g);
                    for (const attrMatch of attrMatches) {
                        attributes[attrMatch[1]] = attrMatch[2];
                    }
                    
                    return { tag, attributes };
                };
                
                const parsed1 = parseSignature(sig1);
                const parsed2 = parseSignature(sig2);
                
                if (!parsed1 || !parsed2 || parsed1.tag !== parsed2.tag) {
                    return false;
                }
                
                const attrs1 = parsed1.attributes;
                const attrs2 = parsed2.attributes;
                
                // Check whether attribute keys match.
                const keys1 = Object.keys(attrs1).sort();
                const keys2 = Object.keys(attrs2).sort();
                
                if (keys1.length !== keys2.length) {
                    return false;
                }
                
                // Require same keys but allow value differences.
                for (let i = 0; i < keys1.length; i++) {
                    if (keys1[i] !== keys2[i]) {
                        return false;
                    }
                    
                    const key = keys1[i];
                    const val1 = attrs1[key];
                    const val2 = attrs2[key];
                    
                    // For selected keys, values must be equal.
                    if (['class', 'role', 'type', 'method'].includes(key)) {
                        if (val1 !== val2) {
                            return false;
                        }
                    }
                    
                    // For others (href/id/aria-label, etc.), values may differ
                    // to allow per-item dynamic content.
                }
                
                return true;
            }
            
            // Helper: get internal structure of child element (relative XPath list).
            function getInternalStructure(childElement) {
                const structure = [];
                
                Array.from(childElement.children).forEach(child => {
                    const relativePath = getRelativeXPath(child, childElement);
                    structure.push(relativePath);
                });
                
                return structure.sort(); // Sort for stable comparison.
            }
            
            // Main detection function.
            function detectDynamicLists() {
                const results = [];
                
                // Step 1: find all potential containers.
                const potentialContainers = [];
                document.querySelectorAll('*').forEach(element => {
                    const directChildren = Array.from(element.children).filter(child => 
                        child.nodeType === 1 && // Element nodes only.
                        getComputedStyle(child).display !== 'none' && // Visible elements only.
                        !child.hidden
                    );
                    
                    if (directChildren.length >= MIN_ITEMS) {
                        potentialContainers.push({
                            element: element,
                            children: directChildren
                        });
                    }
                });
                
                console.log(`Found ${potentialContainers.length} potential containers`);
                
                // Step 2: check whether direct children are structurally similar.
                const attributeFiltered = [];
                potentialContainers.forEach((container, index) => {
                    const children = container.children;
                    
                    if (children.length === 0) return;
                    
                    // Get attribute signatures for all child elements.
                    const signatures = children.map(child => getElementAttributeSignature(child));
                    
                    // Check whether all child elements are structurally similar.
                    const firstSignature = signatures[0];
                    const allSimilar = signatures.every(sig => 
                        areAttributeSignaturesSimilar(sig, firstSignature)
                    );
                    
                    if (allSimilar) {
                        attributeFiltered.push({
                            element: container.element,
                            children: children,
                            childSignature: firstSignature,
                            allSignatures: signatures // Keep all signatures for debugging.
                        });
                    } else {
                        // Debug info: print mismatch reason.
                        console.log(`Container rejected due to attribute mismatch:`, {
                            container: container.element.tagName + (container.element.className ? '.' + container.element.className : ''),
                            signatures: signatures.slice(0, 3) // Show first 3 only.
                        });
                    }
                });
                
                console.log(`After attribute filtering: ${attributeFiltered.length} containers`);
                
                // Step 3: check whether child internal structures are identical.
                const structureFiltered = [];
                attributeFiltered.forEach(container => {
                    const children = container.children;
                    
                    if (children.length === 0) return;
                    
                    // Use first child internal structure as template.
                    const templateStructure = getInternalStructure(children[0]);
                    
                    // Verify all child elements share the same internal structure.
                    const allMatch = children.every(child => {
                        const childStructure = getInternalStructure(child);
                        
                        // Compare structure arrays.
                        if (childStructure.length !== templateStructure.length) {
                            return false;
                        }
                        
                        return childStructure.every((path, idx) => 
                            path === templateStructure[idx]
                        );
                    });
                    
                    if (allMatch) {
                        structureFiltered.push({
                            element: container.element,
                            children: children,
                            childSignature: container.childSignature,
                            internalStructure: templateStructure,
                            allSignatures: container.allSignatures
                        });
                    }
                });
                
                console.log(`After structure filtering: ${structureFiltered.length} containers`);
                
                // Step 4: build final output.
                const finalResults = [];
                structureFiltered.forEach(container => {
                    try {
                        const containerXPath = getFullXPath(container.element);
                        
                        finalResults.push({
                            container_xpath: containerXPath,
                            child_tag_attributes: container.childSignature,
                            child_internal_structure: container.internalStructure,
                            item_count: container.children.length,
                            // Additional debug fields.
                            container_tag: container.element.tagName.toLowerCase(),
                            container_id: container.element.id || '',
                            container_classes: Array.from(container.element.classList),
                            // Show child signature variation samples (for debugging).
                            signature_variations: container.allSignatures.slice(0, 3) // First 3 signatures.
                        });
                    } catch (error) {
                        console.warn('Error generating result for container:', error);
                    }
                });
                
                // Sort by DOM order (XPath-based).
                finalResults.sort((a, b) => {
                    return a.container_xpath.localeCompare(b.container_xpath);
                });
                
                console.log(`Final results: ${finalResults.length} dynamic lists detected`);
                return finalResults;
            }
            
            // Run detection and return results.
            try {
                return detectDynamicLists();
            } catch (error) {
                console.error('Error in dynamic list detection:', error);
                return [];
            }
        }
        """
        
        try:
            results = page.evaluate(js_code)
            return results if results else []
        except Exception as e:
            print(f"Error executing dynamic list detection: {e}")
            return []
    
    async def get_container_details(self, page: Page, container_xpath: str) -> Dict[str, Any]:
        """
        Get detailed information for the specified container.
        
        Args:
            page: Playwright page object
            container_xpath: Container XPath
            
        Returns:
            Dict: Container detail dictionary
        """
        
        js_code = """
        () => {
            const xpath = `"""+container_xpath+"""`;
            
            try {
                const result = document.evaluate(
                    xpath, 
                    document, 
                    null, 
                    XPathResult.FIRST_ORDERED_NODE_TYPE, 
                    null
                );
                
                const container = result.singleNodeValue;
                if (!container) {
                    return { error: 'Container not found' };
                }
                
                const children = Array.from(container.children).filter(child => 
                    child.nodeType === 1 && 
                    getComputedStyle(child).display !== 'none' && 
                    !child.hidden
                );
                
                return {
                    exists: true,
                    tag: container.tagName.toLowerCase(),
                    id: container.id || '',
                    classes: Array.from(container.classList),
                    child_count: children.length,
                    sample_child_html: children.length > 0 ? children[0].outerHTML.substring(0, 200) + '...' : '',
                    visible: getComputedStyle(container).display !== 'none' && !container.hidden
                };
                
            } catch (error) {
                return { error: error.message };
            }
        }
        """
        
        try:
            return page.evaluate(js_code)
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        self.cdp.detach()

    def get_screenshot(self):
        return self.page.screenshot(full_page=True)
