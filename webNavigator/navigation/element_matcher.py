import re
import json
from typing import List, Dict, Any, Optional, Union
import urllib


class DOMElementMatcher:
    """DOM element matcher driven by JSON configuration."""
    
    def __init__(self):
        self.config = None
    
    def load_config(self, config: Union[str, List[Dict]]):
        """Load matcher configuration.
        
        Args:
            config: JSON string or parsed configuration dict list.
        """
        if isinstance(config, str):
            self.config = json.loads(config)
        else:
            self.config = config
    
    def match_candidates(self, candidates: List[Dict[str, Any]], domain: str = None) -> Dict[str, List[Dict]]:
        """Match candidate elements.
        
        Args:
            candidates: Candidate element list, each item includes xpath/href/etc.
            domain: Current domain used to match block_domain.
            
        Returns:
            Match result dict: key is element name, value is matched candidates.
        """
        if not self.config:
            raise ValueError("Please load config first")
        
        results = {}
        
        for domain_config in self.config:
            # Check domain matching.
            if domain and domain_config.get("block_domain"):
                if not self._match_domain(domain, domain_config["block_domain"]):
                    continue
            
            # Process each block_element.
            for element_config in domain_config.get("block_elements", []):
                element_name = element_config["name"]
                patterns = element_config["patterns"]
                
                matched_candidates = []
                for candidate in candidates:
                    if self._match_patterns(candidate, patterns):
                        matched_candidates.append(candidate)
                
                if element_name not in results:
                    results[element_name] = []
                results[element_name].extend(matched_candidates)
        
        return results
    
    def _match_domain(self, current_domain: str, pattern_domain: str) -> bool:
        def get_hostname(url: str) -> Optional[str]:
            """Extract the hostname component from a URL string."""
            if not url:
                return None

            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme and not parsed.netloc:
                parsed = urllib.parse.urlparse(f"//{url}", scheme="http")

            return parsed.netloc
        current_domain = get_hostname(current_domain)
        pattern_domain = get_hostname(pattern_domain)
        """Match domain."""
        # Simple domain matching; can be extended to regex-based matching.
        return current_domain == pattern_domain
    
    def _match_patterns(self, candidate: Dict[str, Any], patterns: List[Dict]) -> bool:
        """Match pattern list."""
        if not patterns:
            return True
        
        results = []
        current_logic = "AND"
        
        for pattern in patterns:
            # Get the logical operator of current pattern.
            pattern_logic = pattern.get("logic", "AND")
            
            # Match current pattern.
            pattern_result = self._match_single_pattern(candidate, pattern)
            
            # Handle NOT logic.
            if pattern_logic == "NOT":
                pattern_result = not pattern_result
            
            results.append((pattern_result, pattern_logic))
        
        # Evaluate final result.
        return self._evaluate_logic_results(results)
    
    def _match_single_pattern(self, candidate: Dict[str, Any], pattern: Dict) -> bool:
        """Match a single pattern."""
        for key, value in pattern.items():
            if key == "logic":
                continue
            
            if key == "xpath":
                if not self._match_regex(candidate.get("xpath", ""), value):
                    return False
            elif key == "href":
                if not self._match_regex(candidate.get("href", ""), value):
                    return False
            else:
                # Match other attributes.
                candidate_value = candidate.get(key, "")
                if not self._match_regex(candidate_value, value):
                    return False
        
        return True
    
    def _match_regex(self, candidate_value: str, pattern: str) -> bool:
        """Regex match helper."""
        if not pattern:
            return not candidate_value
        
        try:
            return bool(re.search(pattern, candidate_value or ""))
        except re.error:
            # If pattern is not valid regex, do plain string match.
            return candidate_value == pattern
    
    def _evaluate_logic_results(self, results: List[tuple]) -> bool:
        """Evaluate logical expression results."""
        if not results:
            return True
        
        # Evaluate left-to-right.
        final_result = results[0][0]
        
        for i in range(1, len(results)):
            current_result, current_logic = results[i]
            prev_logic = results[i-1][1] if i > 0 else "AND"
            
            if prev_logic == "AND":
                final_result = final_result and current_result
            elif prev_logic == "OR":
                final_result = final_result or current_result
        
        return final_result


# Usage example.
if __name__ == "__main__":
    # Example config.
    config = [
        {
            "block_domain": "gitlab.com",
            "block_elements": [
                {
                    "name": "element1",
                    "patterns": [
                        {
                            "xpath": "/html/body/section/div/form/fieldset/div",
                            "logic": "AND"
                        },
                        {
                            "href": "https://.*\\.com",
                            "logic": "NOT"
                        },
                        {
                            "class": "form-.*",
                            "logic": "AND"
                        }
                    ]
                },
                {
                    "name": "element2",
                    "patterns": [
                        {
                            "xpath": "/html/body/section/div/form/fieldset/div[3:]",
                            "logic": "AND"
                        }
                    ]
                }
            ]
        }
    ]
    
    # Example candidate elements.
    candidates = [
        {
            "xpath": "/html/body/section/div/form/fieldset/div[1]/input",
            "href": "https://example.org/page1",
            "class": "form-input"
        },
        {
            "xpath": "/html/body/section/div/form/fieldset/div[2]/button",
            "href": "",
            "class": "form-button"
        },
        {
            "xpath": "/html/body/section/div/form/fieldset/div[5]/span",
            "href": "",
            "class": "form-text"
        },
        {
            "xpath": "/html/body/header/nav/a",
            "href": "https://gitlab.com/home",
            "class": "nav-link"
        }
    ]
    
    # Create matcher and test.
    matcher = DOMElementMatcher()
    matcher.load_config(config)
    
    results = matcher.match_candidates(candidates, "gitlab.com")
    
    print("Match results:")
    for element_name, matched_elements in results.items():
        print(f"\n{element_name}: {len(matched_elements)} matches")
        for element in matched_elements:
            print(f"  - {element['xpath']}")
    
    # Test loading config from JSON string.
    config_json = json.dumps(config)
    matcher2 = DOMElementMatcher()
    matcher2.load_config(config_json)
    
    results2 = matcher2.match_candidates(candidates, "gitlab.com")
    print(f"\nConsistent results when loading from JSON string: {results == results2}")