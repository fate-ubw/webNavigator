import re
import copy
import os
from typing import Any, Dict
from functools import partial
import random
import json
from typing import Any, TypedDict
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


from webNavigator.obs_opt import parse_node_descendants, parse_node_ancestors, parse_node_siblings, action_set_invisible, action_set_visible, action_set_visible_if_with_name, translate_node_to_str, construct_new_DOM_with_visible_nodes
from webNavigator.llms.claude import call_claude, call_claude_with_messages, arrange_message_for_claude
from webNavigator.llms.mistral import call_mistral, call_mistral_with_messages, arrange_message_for_mistral
from webNavigator.llms.cohere import call_cohere, call_cohere_with_messages, arrange_message_for_cohere
from webNavigator.llms.llama import call_llama, call_llama_with_messages, arrange_message_for_llama
from webNavigator.llms.titan import call_titan, call_titan_with_messages, arrange_message_for_titan
from webNavigator.llms.gpt import call_gpt, call_gpt_with_messages, arrange_message_for_gpt
from webNavigator.llms.gemini import call_gemini, call_gemini_with_messages, arrange_message_for_gemini
from webNavigator.utils import CURRENT_DIR
from webNavigator import DomainNavigator
from webNavigator.utils import find_image_by_name, json_parser
from webNavigator.prompts import page_selector

DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS = ["observation", "action"]
DEFAULT_ONLINE_INTERACTION_ELEMENTS = ["url", "observation"]
MODEL_FAMILIES = ["claude", "mistral", "cohere", "llama", "titan", "gpt", "gemini"]
CALL_MODEL_MAP = {
    "claude": call_claude,
    "mistral": call_mistral,
    "cohere": call_cohere,
    "llama": call_llama,
    "titan": call_titan,
    "gpt": call_gpt,
    "gemini": call_gemini,
}
CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP = {
    "claude": call_claude_with_messages,
    "mistral": call_mistral_with_messages,
    "cohere": call_cohere_with_messages,
    "llama": call_llama_with_messages,
    "titan": call_titan_with_messages,
    "gpt": call_gpt_with_messages,
    "gemini": call_gemini_with_messages,
}
ARRANGE_MESSAGE_FOR_MODEL_MAP = {
    "claude": arrange_message_for_claude,
    "mistral": arrange_message_for_mistral,
    "cohere": arrange_message_for_cohere,
    "llama": arrange_message_for_llama,
    "titan": arrange_message_for_titan,
    "gpt": arrange_message_for_gpt,
    "gemini": arrange_message_for_gemini,
}


DEFAULT_DOMAIN_CONFIG = {
    "topk": 25,
    "return_multivector": True,
    "paths": {
        "shopping_admin": "webNavigator/webNodes/run_id-20251026-151146-admin",
        "map": "webNavigator/webNodes/run_id-20251111-180558-map",
        "shopping": "webNavigator/webNodes/run_id-20251107-133622-shopping",
        "gitlab": "webNavigator/webNodes/run_id-20251026-190417-gitlab",
        "reddit": "webNavigator/webNodes/run_id-20251207-190950-postmill",
    }
}

class PlanTreeNode:
    def __init__(self, id, type, text, level, url, step):
        self.visible = True
        self.id = id
        self.type = type
        self.text = text
        self.level = level
        self.url = url
        self.step = step
        self.children = []
        self.parent = None
        self.note = []
        self.hint = []
        self.resume_reason = []
        self.steps_taken = []

    def reset(self):
        self.visible = True
        self.note = []
        self.hint = []
        self.steps_taken = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def search_node_by_id(self, target_id):
        if self.visible and self.id == target_id:
            return self
        for child in self.children:
            result = child.search_node_by_id(target_id)
            if result:
                return result
        return None
    
    def traverse(self, action=None, tree_buffer=[]):
        res_action = action(self)
        if res_action:
            if isinstance(res_action, list):
                tree_buffer.extend(res_action)
            else:
                tree_buffer.append(res_action)
        for child in self.children:
            child.traverse(action, tree_buffer=tree_buffer)

class Agent:
    def __init__(self, config, objective, prompt_template):
        self.config = config
        self.objective = objective
        self.prompt_template = prompt_template

        if hasattr(self.config, "documented_interaction_elements"):
            self.previous_interactions = {k: [] for k in set(DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS+self.config.documented_interaction_elements)}
        else:
            self.previous_interactions = {k: [] for k in DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS}
        if hasattr(self.config, "online_interaction_elements"):
            self.online_interaction = {k: None for k in set(DEFAULT_ONLINE_INTERACTION_ELEMENTS+self.config.online_interaction_elements)}
        else:
            self.online_interaction = {k: None for k in DEFAULT_ONLINE_INTERACTION_ELEMENTS}

        # Allow overriding model family via config (e.g., proxy API setups).
        if hasattr(self.config, "model_family"):
            self.model_family = self.config.model_family
        else:
            self.model_family = [model_family for model_family in MODEL_FAMILIES if model_family in self.config.model][0]
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=self.config.model)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=self.config.model)
        self.call_page_selector_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=self.config.selector.model)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

    def shift_model(self, model_id):
        self.model_family = [model_family for model_family in MODEL_FAMILIES if model_family in model_id][0]
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=model_id)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=model_id)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

    def prune_message_list(self, message_list):
        return self.merge_adjacent_text([m for m in message_list if not (m[0]=="text" and len(m[1])==0)])
    
    def merge_adjacent_text(self, message_list):
        merged_list = []
        current_tuple = None
        
        for tup in message_list:
            if tup[0] == "text":
                if current_tuple:
                    current_tuple = (current_tuple[0], current_tuple[1] + tup[1])
                else:
                    current_tuple = tup
            else:
                if current_tuple:
                    merged_list.append(current_tuple)
                    current_tuple = None
                merged_list.append(tup)
        
        if current_tuple:
            merged_list.append(current_tuple)
        
        return merged_list

    
    def get_step(self):
        return len(self.previous_interactions["action"])

    def update_objective(self, objective):
        self.objective = objective

    def update_online_state(self, **online_states):
        for k in online_states.keys():
            if k in self.online_interaction.keys():
                self.online_interaction[k] = online_states[k]

    def update_history(self, **interaction_dict):
        for k in interaction_dict.keys():
            if k in self.previous_interactions.keys():
                self.previous_interactions[k].append(interaction_dict[k])

    def equal_history_length(self):
        lengths = [len(self.previous_interactions[k]) for k in self.previous_interactions.keys()]
        return (len(set(lengths)) == 1)

    def parse_elements(self, text, key_list):
        element_dict = {}
        for k in key_list:
            # _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\d\s\W]*: *\n|$)', text, re.DOTALL)
            _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\s]*:|$)', text, re.DOTALL)
            element_dict[k] = _match.group(1).strip() if _match else ""
        return element_dict

    def get_output_specifications(self):
        output_specifications = "\n".join([f"{o.upper()}:\n" + "".join(open(os.path.join(CURRENT_DIR, "webNavigator", "prompts", "output_specifications", "{}.txt".format(o.replace(" ", "_"))), "r").readlines()) for o in self.config.output])
        return output_specifications

    def parse_stipulated_action_list(self, text: str, action: str, actions: list) -> str:
        pattern = rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)'
        return [match[0].strip() for match in re.findall(pattern, text, re.DOTALL)]

    def parse_str_to_action_list(self, text:str, actions: list):
        remain_text = copy.deepcopy(text)
        action_list = []
        while remain_text:
            find_action = False
            for action in actions:
                if remain_text.startswith(action):
                    match = re.search(rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)', remain_text, re.DOTALL)
                    action_list.append(match[0])
                    remain_text = remain_text[len(match[0]):].strip()
                    find_action = True
            if not find_action:
                break
        return action_list

    def get_observation_text(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["text"]
            return self.online_interaction["observation"]["text"]
        elif isinstance(self.online_interaction["observation"], str):
            if idx:
                return self.previous_interactions["observation"][idx]
            return self.online_interaction["observation"]

    def get_observation_image(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["image"]
            return self.online_interaction["observation"]["image"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node"]
            return self.online_interaction["observation"]["node"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node_str(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node_str"]
            return translate_node_to_str(self.online_interaction["observation"]["node"], mode="name_only")
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def del_observation_node(self): # Release historical DOM trees and keep compact string snapshots.
        if isinstance(self.online_interaction["observation"], str):
            return
        if isinstance(self.online_interaction["observation"], dict):
            for idx in range(len(self.previous_interactions["observation"])):
                if "node" in self.previous_interactions["observation"][idx].keys() and self.previous_interactions["observation"][idx]["node"]:
                    node_str = translate_node_to_str(self.previous_interactions["observation"][idx]["node"], mode="name_only")
                    self.previous_interactions["observation"][idx]["node_str"] = node_str
                    self.previous_interactions["observation"][idx]["node"].delete_tree()
                    self.previous_interactions["observation"][idx]["node"] = None


class Actor(Agent):
    def __init__(self, config, objective, prompt_template, plan_tree_node, required_sites=None):
        super().__init__(config, objective, prompt_template)
        self.config = config
        self.plan_tree_root = plan_tree_node
        self.active_node = plan_tree_node
        self.output_specifications = None
        self.planning_specifications = None
        self.navigation_specifications = None
        self.criticism_element_list = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")
        self.domain_navigators: dict[str, DomainNavigator | None] = {}
        
        self.domain_paths = {
            "shopping_admin": getattr(config.domain_paths, "shopping_admin", DEFAULT_DOMAIN_CONFIG["paths"]["shopping_admin"]),
            "map": getattr(config.domain_paths, "map", DEFAULT_DOMAIN_CONFIG["paths"]["map"]),
            "shopping": getattr(config.domain_paths, "shopping", DEFAULT_DOMAIN_CONFIG["paths"]["shopping"]),
            "gitlab": getattr(config.domain_paths, "gitlab", DEFAULT_DOMAIN_CONFIG["paths"]["gitlab"]),
            "reddit": getattr(config.domain_paths, "reddit", DEFAULT_DOMAIN_CONFIG["paths"]["reddit"]),
            "www.google.com": "webNavigator/webNodes/online_mind2web/explore_results-cutted-corp_3000/run_id-20260109-161211-scenario_mind2web_84_www_google_com-depth_1",
            "google.com": "webNavigator/webNodes/online_mind2web/explore_results-cutted-corp_3000/run_id-20260109-161211-scenario_mind2web_84_www_google_com-depth_1",
            "google": "webNavigator/webNodes/online_mind2web/explore_results-cutted-corp_3000/run_id-20260109-161211-scenario_mind2web_84_www_google_com-depth_1",
        } 
        self.required_sites = required_sites
        self._auto_discover_missing_domains(required_sites)
        self._init_domains(required_sites=required_sites) # init domain navigators based on required sites
    
    def _auto_discover_missing_domains(self, required_sites=None) -> None:
        """Auto-discover domain paths for required sites that are not manually configured"""
        if required_sites is None:
            return
        
        webNodes_root = os.path.join(CURRENT_DIR, "webNavigator", "webNodes")
        if not os.path.exists(webNodes_root):
            print(f"[AutoDiscovery] webNodes directory not found: {webNodes_root}")
            return
        
        for site in required_sites:
            # Skip if already configured
            if site in self.domain_paths and self.domain_paths[site] is not None:
                continue
            
            # Convert site name for matching: www.traderjoes.com -> www_traderjoes_com
            site_pattern = site.replace(".", "_")
            
            # Recursively search for matching directories
            found_path = self._search_domain_path(webNodes_root, site_pattern)
            
            if found_path:
                self.domain_paths[site] = found_path
                print(f"[AutoDiscovery] Found domain path for '{site}': {found_path}")
            else:
                self.domain_paths[site] = None
                print(f"[AutoDiscovery] Warning: No matching directory found for site '{site}' (pattern: {site_pattern})")
    
    def _search_domain_path(self, root_dir: str, site_pattern: str) -> str | None:
        """Recursively search for directory containing the site pattern"""
        try:
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    # Check if current directory name contains the pattern
                    if site_pattern in item:
                        return item_path
                    
                    # Recursively search subdirectories
                    result = self._search_domain_path(item_path, site_pattern)
                    if result:
                        return result
        except (OSError, PermissionError) as e:
            print(f"[AutoDiscovery] Error accessing directory {root_dir}: {e}")
        
        return None
    
    def _init_domains(self, required_sites=None) -> None:
        """Initialize DomainNavigators based on required sites, or all if not specified"""
        # If required_sites is provided, only initialize those domains (lazy loading)
        # If not provided, initialize all domains (backward compatible)
        if required_sites is None:
            domains_to_init = self.domain_paths.keys()
            print(f"[NavigatorReactAgent] No required_sites specified, initializing all domains")
        else:
            domains_to_init = required_sites
            print(f"[NavigatorReactAgent] Initializing only required domains: {required_sites}")

        for domain_name in domains_to_init:
            if domain_name not in self.domain_paths:
                print(f"[NavigatorReactAgent] Warning: '{domain_name}' not in domain_paths")
                self.domain_navigators[domain_name] = None
                continue
            
            domain_path = self.domain_paths[domain_name]
            if domain_path and os.path.exists(domain_path):
                print(f"[NavigatorReactAgent] Initializing DomainNavigator for '{domain_name}' from {domain_path}")
                self.domain_navigators[domain_name] = DomainNavigator(domain_name, domain_path, self.config)
            else:
                print(f"[NavigatorReactAgent] Warning: domain path not found for '{domain_name}': {domain_path}")
                self.domain_navigators[domain_name] = None
        
        # Initialize remaining domains as None if they weren't in required_sites
        if required_sites is not None:
            for domain_name in self.domain_paths.keys():
                if domain_name not in self.domain_navigators:
                    self.domain_navigators[domain_name] = None

    def update_online_state(self, **online_states):
        super().update_online_state(**online_states)

    def is_planning(self, action):
        for c in self.config.planning_command:
            if action.startswith(c):
                return c
        return False

    def is_navigation(self, action):
        action_without_note = re.sub(rf'(note\s*(.*?))(?=\n(?:{"|".join(map(re.escape, self.config.navigation_command))})|$)', "", action).strip()
        for c in self.config.navigation_command:
            if action_without_note.startswith(c):
                return c
        return False
    
    def is_valid_action(self, action_str):
        action = (
            action_str.split("[")[0].strip()
            if "[" in action_str
            else action_str.split()[0].strip()
        )
        match action:
            case "optimal_navigate":
                match = re.search(r"optimal_navigate ?\[([\s\S]+?)\]\s*\[([\s\S]+?)\]\s*\[([\s\S]+?)\]", action_str)
                if not match:
                    return False
                think = match.group(1).strip()
                domain = match.group(2).strip()
                query = match.group(3).strip()
                if think and domain and query:
                    return True
                return False
            case "click":
                match = re.search(r"click ?\[(\d+)\]", action_str)
                if not match:
                    return False
                element_id = match.group(1)
                if element_id in self.get_observation_text():
                    return True
                return False
            case "type":
                if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                    action_str += " [1]"

                match = re.search(
                    r"type ?\[(\d+)\] ?\[(.*)\] ?\[(\d+)\]", action_str, re.DOTALL
                )
                if not match:
                    return False
                element_id, text, enter_flag = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                )
                enter_flag = True if enter_flag == "1" else False
                if enter_flag:
                    text += "\n"
                if element_id in self.get_observation_text():
                    return True
            case "go_back":
                return True
            case "note":
                return True
            case "stop":
                return True
            case "goto":
                return True
            case "scroll":
                return True

    def are_valid_actions(self, actions):
        action_list = self.parse_str_to_action_list(actions, self.config.planning_command+self.config.navigation_command+["goto"])
        if not action_list:
            return False
        for action in action_list:
            if not self.is_valid_action(action):
                return False
        return True

    def get_previous_plans(self, verbose=False):
        def action_return_visible_node(node, verbose=False):
            if node.id == self.active_node.id:
                basic = "\t" * node.level + f"[{node.id}] (Active Plan) {node.text}" if node.visible else None
            else:
                basic = "\t" * node.level + f"[{node.id}] {node.text}" if node.visible else None
            if basic and len(node.resume_reason) > 0:
                basic += f" # Was resumed to this step {len(node.resume_reason)} time(s) for:"
                for i, reason in enumerate(node.resume_reason):
                    basic += f" {i}. {reason}"
            if verbose and basic and len(node.note) > 0:
                for i, note in enumerate(node.note):
                    basic += "\n" + "\t" * node.level + f"Note {i}. {note}"
            return basic
        plan_tree_buffer = []
        parse_node_descendants(self.plan_tree_root, partial(action_return_visible_node, verbose=verbose), tree_buffer=plan_tree_buffer)
        return "\n".join(plan_tree_buffer)
    
    def get_active_plan(self):
        return f"[{self.active_node.id}] {self.active_node.text}"
    
    def get_interaction_history(self, interaction_history_config=False, mode="highlight"):
        interaction_history_config = interaction_history_config if interaction_history_config else self.config.interaction_history

        previous_observation = []
        for i in self.active_node.steps_taken:
            if self.get_observation_node_str() and self.get_observation_node_str(i) and not self.get_observation_node_str() == self.get_observation_node_str(i):
                if self.previous_interactions["observation highlight"][i] and mode == "highlight" and len(translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]).split()) < 200:
                    try:
                        previous_observation.append({"text": translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]), "image": self.get_observation_image(i)})
                    except:
                        print(i, self.previous_interactions["observation"][i]["text"])
                        raise ValueError("Cannot translate highlight node to text.")
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            elif not self.get_observation_node() or mode == "full":
                if len(self.get_observation_text(i).split()) < 200:
                    previous_observation.append({"text": self.get_observation_text(i), "image": self.get_observation_image(i)})
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            else:
                previous_observation.append({"text": "The same as the CURRENT OBSERVATION (see below CURRENT OBSERVATION section).", "image": self.get_observation_image(i)})

        previous_observation_summary = [self.previous_interactions["observation summary"][i] for i in self.active_node.steps_taken]

        def get_text(obs):
            if isinstance(obs, dict):
                return obs["text"]
            elif isinstance(obs, str):
                return obs

        def get_image(obs):
            if isinstance(obs, dict):
                return obs["image"]
            elif isinstance(obs, str):
                return obs

        if interaction_history_config.step_num == "all":
            textual_observations = [get_text(obs) for obs in previous_observation] if interaction_history_config.verbose else previous_observation_summary
            visual_observations = [get_image(obs) for obs in previous_observation]
        else:
            textual_observations = previous_observation_summary[:-interaction_history_config.step_num]
            visual_observations = [None] * len(previous_observation_summary[:-interaction_history_config.step_num])
            textual_observations += [get_text(obs) for obs in previous_observation][-interaction_history_config.step_num:] if interaction_history_config.verbose else previous_observation_summary[-interaction_history_config.step_num:]
            visual_observations += [get_image(obs) for obs in previous_observation][-interaction_history_config.step_num:]

        plans = [self.previous_interactions["plan"][i] for i in self.active_node.steps_taken]
        reasons = [self.previous_interactions["reason"][i] for i in self.active_node.steps_taken]
        actions = [self.previous_interactions["action"][i] for i in self.active_node.steps_taken]
        mode_analyses = [self.previous_interactions["mode analysis"][i] for i in self.active_node.steps_taken] 
        exact_modes = ["EXPLORATION" if "EXPLORATION" in mode else ("LOW-LEVEL ACTION" if " ACTION" in mode else mode) for mode in mode_analyses]
        current_url = [self.previous_interactions["url"][i] for i in self.active_node.steps_taken]
        if "image" in interaction_history_config.type:
            message_list = []
            for step, (obs, vi_obs, plan, reason, action) in enumerate(zip(textual_observations, visual_observations, plans, reasons, actions)):
                message_list.append(("text", f"<step_{step}_interaction>\n"))
                if vi_obs is not None:
                    message_list.append(("text", "VISUAL OBSERVATION:\n"))
                    message_list.append(("image", vi_obs))
                if self.active_node.id != 0:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
                else:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
            return self.prune_message_list(message_list=message_list)
        else:
            message = ""
            for step, (obs, plan, reason, action, current_url, mode) in enumerate[tuple[Any | str | None, Any, Any, Any, Any, Any]](zip(textual_observations, plans, reasons, actions, current_url, exact_modes)):
                if "optimal_navigate [" in action:
                    if self.previous_interactions["target_page"][step] is None:
                        action = f'''[NAVIGATION FAILED] Used optimal_navigate tool.\n
                        Think: "{self.previous_interactions["navigate_think"][step]}"\n
                        Query: "{self.previous_interactions["navigate_query"][step]}"\n
                        Domain: {self.previous_interactions["navigate_domain"][step]}\n
                        Result: No matching page found in the knowledge base for this query. The optimal_navigate strategy has been exhausted.
                        Status: Exploration phase failed. Upon the failure of high-level navigation strategies, you are to pivot to low-level actions (e.g., click, type) on the current page; specifically, for tasks requiring the discovery of concrete entities like products, projects, users, or orders, you must utilize the page's search bar or advanced search functionality.
                        '''
                        # Page Selection Reasoning: {self.previous_interactions["page_selection_response"][step]['reasoning']}\n # add Page selector Reasoing will cause issuing optimal_navigate tool repeatedly.
                    else:
                        action = f'''[NAVIGATION SUCCESS] Used optimal_navigate tool.\n
                        Think: "{self.previous_interactions["navigate_think"][step]}"\n
                        Query: "{self.previous_interactions["navigate_query"][step]}"\n
                        Domain: {self.previous_interactions["navigate_domain"][step]}\n
                        Result: Successfully teleported to target page "{self.previous_interactions["target_page"][step]}".\n
                        Status: Exploration phase successful. Now use other low-level actions (e.g., click, type) on the current page to complete the task.
                        '''
                        # Page Selection Reasoning: {self.previous_interactions["page_selection_response"][step]['reasoning']}\n
                if self.active_node.id != 0:
                    message += f"<step_{step}_interaction>\nCURRENT_PAGE_URL:{current_url}\nMODE_ANALYSIS:{mode}\nOBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
                else:
                    message += f"<step_{step}_interaction>\nCURRENT_PAGE_URL:{current_url}\n\nMODE_ANALYSIS:{mode}\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n" # f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
            return self.prune_message_list(message_list=[("text", message)])
        
    def pre_process_atomic_actions(self, atomic_action_list=["combobox"]):
        if self.get_observation_node() and "combobox" in atomic_action_list:
            self.online_interaction["observation"]["text"] = translate_node_to_str(self.get_observation_node(), mode="concise", hidden_roles=["menu", "combobox", "listbox"])

    def get_online_input(self, criticism_elements):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        # reddit: fix website bug in reddit
        source_obs = self.get_observation_text()
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "interaction history": self.get_interaction_history(), #
            "current observation": source_obs,
            "current visual observation": self.get_observation_image()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            elif input_type.startswith("critic: ") and criticism_elements and input_type[len("critic: "):] in criticism_elements.keys() and criticism_elements[input_type[len("critic: "):]]:
                input_type = input_type[len("critic: "):]
                input_content = criticism_elements[input_type]
                input_type = "FROM USER: " + input_type
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]
            elif input_type == "step" and isinstance(input_content, int):
                input_list.append(("text", f"CURRENT {input_type.upper()}:\n{input_content}\n"))

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])
    
    def get_planning_specifications(self):
        if self.planning_specifications:
            return self.planning_specifications
        self.planning_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "webNavigator", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
        return self.planning_specifications
    
    def get_navigation_specifications(self):
        if self.navigation_specifications:
            return self.navigation_specifications
        self.navigation_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "webNavigator", "prompts", "navigation_specifications", f"{n}.txt"), "r").readlines()) for n in self.config.navigation_command])
        return self.navigation_specifications
    
    def get_actor_instruction(self, examples=None):
        instruction = self.prompt_template["instruction_template"]["without_planning"]#TODO:remove discarded prompt
        output_specifications = self.get_output_specifications()
        planning_specifications = self.get_planning_specifications()
        navigation_specifications = self.get_navigation_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", planning_specifications)
        instruction = instruction.replace("{navigation_specifications}", navigation_specifications)

        example_source = examples if examples is not None else self.prompt_template.get("examples", [])
        if len(example_source) > 0:
            instruction += f"\n\n## Here are a few examples:"
            for i, example in enumerate(example_source):
                example_input = example["input"]
                example_output = example["output"]
                if "example_template" in self.prompt_template.keys():
                    instruction += "\n\n"
                    instruction += self.prompt_template.get("example_template", "| Example {i}\n### Input:\n{example_input}\n### Response: Let's think step by step.\n{example_response}").replace("{i}", i).replace("{example_input}", example_input).replace("{example_output}", example_output)
                else:
                    instruction += f"\n\n| Example {i}\n\n### Input:\n{example_input}\n\n### Response: Let's think step by step.\n{example_output}"
        
        if self.get_step() == self.config.others.max_steps - 1:
            instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and this would be your FINAL STEP. Wrap up your observations and return your answer with `stop [answer]` to maximize the reward."
        # else:
        #     instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and there are {self.config.others.max_steps-self.get_step()} remaining attempts."

        return instruction
    
    def verbose(self, instruction, online_input, model_response_list, action_element_list):
        action_element_keys = [k for k in self.config.play if k in action_element_list[0].keys()]
        other_play_keys = [k for k in self.config.play if k not in action_element_list[0].keys()]

        VERBOSE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "url": self.online_interaction["url"],
            "observation": self.get_observation_text(),
            "response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join([f"|\tAgent {i}:\n{model_response}" for i, model_response in enumerate(model_response_list[:self.config.number])]) if self.config.number > 1 else model_response_list[0],
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
        }

        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"ACTOR"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP.get(t, "")
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")
            with open(self.output_play_path, "w") as _:
                pass
            for p in other_play_keys:
                content = VERBOSE_TO_CONTENT_MAP.get(p, "")
                with open(self.output_play_path, "a") as af:
                    af.write(f"{p.upper()}:\n{content}\n\n")
            for i, action_elements in enumerate(action_element_list):
                if len(action_element_list) > 1:
                    with open(self.output_play_path, "a") as af:
                        af.write("-"*32+f"AGENT {i}"+"-"*32+"\n")
                for action_element_key in action_element_keys:
                    content = action_elements.get(action_element_key, "N/A")
                    with open(self.output_play_path, "a") as af:
                        af.write(f"{action_element_key.upper()}:\n{content}\n\n")
    
    def parse_plan(self, planning):
        planning_type = self.is_planning(action=planning)
        match = re.search(
            rf"{planning_type} ?\[(\d+)\] ?\[(.+)\]", planning, re.DOTALL
        )
        if not match:
            raise ValueError("Invalid planning command.")
        node_id, planning_content = (
            int(match.group(1)),
            match.group(2)
        )
        return planning_type, node_id, planning_content
    
    def prune_planning(self, node:PlanTreeNode, planning_content):
        def set_invisible(node:PlanTreeNode):
            node.visible = False
        def return_steps_taken(node:PlanTreeNode):
            return [node.step] + node.steps_taken
        after_node = False
        if node.id > 0:
            for child in node.parent.children:
                if not after_node and child != node:
                    continue
                elif child == node:
                    after_node = True
                    continue
                child.visible = False
        node.traverse(set_invisible)
        node.reset()
        steps_taken = []
        node.traverse(action=return_steps_taken, tree_buffer=steps_taken)
        node.steps_taken = sorted(list(set(steps_taken)), reverse=False)
        node.resume_reason.append(planning_content)
        navigation = f"goto [{node.url}] [1]"
        self.active_node = node
        return navigation
    
    def branch_planning(self, node, planning_content):
        new_node = PlanTreeNode(id=self.active_node.id+1, type=type, text=planning_content, level=node.level+1, url=self.online_interaction["url"], step=self.get_step())
        self.active_node = new_node
        node.add_child(new_node)
    
    def planning(self, action):
        if action and self.is_planning(action):
            try:
                planning_type, node_id, planning_content = self.parse_plan(planning=action)
                node = self.plan_tree_root.search_node_by_id(node_id)
                if not node:
                    raise ValueError(f"Invalid node id {node_id}: {action}.")
                if planning_type == "prune":
                    navigation_action = self.prune_planning(node=node, planning_content=planning_content)
                    return navigation_action
                elif planning_type == "branch":
                    self.branch_planning(node=node, planning_content=planning_content)
                else:
                    raise ValueError(f"Invalid planning operation {planning_type}: {action}.")
            except Exception as e:
                print("Invalid plan node:", str(e))
                flaw_node = self.active_node
                flaw_node.note.append(f"You previously generate plan \"{action}\", which has INVALID syntax. User planning command like `branch [parent_plan_id] [new_subplan_intent]` or `prune [resume_plan_id] [reason]`.")
        else:
            self.active_node.steps_taken.append(self.get_step())
        return None

    
    def parse_action(self, action_str):
        try:
            DOM_root_node = self.get_observation_node()
            action_str = action_str.strip()
            action = (
                action_str.split("[")[0].strip()
                if "[" in action_str
                else action_str.split()[0].strip()
            )
            match action:
                case "click":
                    match = re.search(r"click ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid click action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"click [{element_id}] ({node.role} {node.name})"
                case "hover":
                    match = re.search(r"hover ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid hover action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"hover [{element_id}] ({node.role} {node.name})"
                case "type":
                    if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                        action_str += " [1]"

                    match = re.search(
                        r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
                    )
                    if not match:
                        raise ValueError(f"Invalid type action {action_str}")
                    element_id, text, enter_flag = (
                        match.group(1),
                        match.group(2),
                        match.group(3),
                    )
                    enter_flag = True if enter_flag == "1" else False
                    if enter_flag:
                        text += "\n"
                    node = DOM_root_node.search_node_by_id(element_id)
                    return action + f" ({node.name})"
                case "scroll":
                    return action_str
                case "goto":
                    return action
                case "new_tab":
                    return action
                case "go_back":
                    return action
                case "go_forward":
                    return action
                case "stop":
                    return action

            return False
        except:
            return False
    
    def parse_actions_to_element_ids(self, actions):
        action_str_list = []
        for a in self.config.navigation_command:
            action_str_list += self.parse_stipulated_action_list(text=actions, action=a, actions=self.config.planning_command+self.config.navigation_command+["goto"])
        retained_element_ids = []
        for action_str in action_str_list:
            try:
                action_str = action_str.strip()
                action = (
                    action_str.split("[")[0].strip()
                    if "[" in action_str
                    else action_str.split()[0].strip()
                )
                match action:
                    case "click":
                        match = re.search(r"click ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid click action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "hover":
                        match = re.search(r"hover ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid hover action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "type":
                        if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                            action_str += " [1]"

                        match = re.search(
                            r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
                        )
                        if not match:
                            raise ValueError(f"Invalid type action {action_str}")
                        element_id, text, enter_flag = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "scroll":
                        pass
                    case "goto":
                        pass
                    case "new_tab":
                        pass
                    case "go_back":
                        pass
                    case "go_forward":
                        pass
                    case "stop":
                        pass
                    case "note":
                        pass

                return retained_element_ids
            except:
                continue

        return retained_element_ids
    
    def take_note(self, action, note_as_action=True):
        if action and "note [" in action:
            none_note_action_list = []
            action_list = self.parse_str_to_action_list(action, actions=self.config.planning_command+self.config.navigation_command+["goto"])
            for a in action_list:
                if "note [" in a:
                    note = re.search(r"note ?\[?(.+)", a, re.DOTALL).group(1)
                    if note.endswith("]"):
                        note = note[:-1]
                    self.active_node.note.append(f"STEP {self.get_step()}: {note}")
                    self.note_buffer = note
                else:
                    none_note_action_list.append(a)
            if note_as_action:
                return action
            return "\n".join(none_note_action_list)
        # action_note = self.parse_action(action)
        # if action_note:
        #     self.active_node.note.append(f"STEP {self.get_step()} ACTION: {action_note}")
        return action
        
    def get_observation_highlight(self, action_elements:dict):
        action_elements["observation highlight idxs"] = copy.deepcopy(action_elements.get("observation highlight", ""))
        DOM_root_node = self.get_observation_node()
        if not DOM_root_node:
            action_elements["observation highlight"] = None
            return
        observation_highlight_idxs = [int(idx.strip()) for idx in action_elements.get("observation highlight", "").split(",") if idx.strip().isdigit()]
        if observation_highlight_idxs:
            parse_node_descendants(node=DOM_root_node, action=action_set_invisible)
            for idx in observation_highlight_idxs:
                try:
                    node = DOM_root_node.search_node_by_id(idx)
                    parse_node_descendants(node=node, action=action_set_visible)
                    parse_node_ancestors(node=node, action=action_set_visible)
                    parse_node_siblings(node=node, action=action_set_visible_if_with_name)
                except:
                    pass
        try: 
            assert DOM_root_node.get_visible_node_number() < 30 and construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            action_elements["observation highlight"] = construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
        except:
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
            action_elements["observation highlight"] = None

        action_elements["retained element ids"] = self.parse_actions_to_element_ids(action_elements["action"])

    def parse_action_from_action_candidates(self, action_elements):
        if "action" in action_elements.keys():
            return action_elements
        assert any("action candidates" in k for k in action_elements.keys())
        action_candidates_key = [k for k in action_elements.keys() if "action candidates" in k][0]
        def parse_reasons_and_actions(input_string):
            pattern = r'- reason: \[(.*?)\]\s*(?:- action: \[(.*?)\])?\s*(?:\n|\Z)'

            matches = re.findall(pattern, input_string, re.DOTALL)

            parsed_data = []
            for match in matches:
                reason = match[0].strip()
                action = match[1].strip()
                if reason and action:
                    parsed_data.append({'reason': reason, 'action': action})

            return parsed_data
        action_elements[action_candidates_key] = parse_reasons_and_actions(action_elements[action_candidates_key])
        return action_elements


    def webpage_selector_messages(
        self, 
        intent: str, 
        query: str, 
        topk_images: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Construct multimodal messages
        
        Args:
            intent: User objective
            query: Search query
            topk_images: Image list, format [{"image_name": "xxx.png", "image_data": "base64..."}, ...]
        
        Returns:
            OpenAI chat messages format list
        """
        # Construct multimodal user content
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"User objective: {intent}\nSearch query: {query}\n\nPlease select the most suitable from the following candidate pages:"}
        ]
        
        for img in topk_images:
            # Add image name
            content.append({"type": "text", "text": f"\nImage name: {img['image_name']}"})
            # Add image
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{img['image_data']}"}
            })
        
        messages = [
            {"role": "user", "content": content}
        ]
        return messages

    def predict_action(self, criticism_elements):
        if self.config.debug > 1:
            action_elements = {k: "" for k in self.config.output}
            human_input = input("ACTION: ")
            action_elements["action"] = human_input
            return [action_elements]
        self.pre_process_atomic_actions() # Hide noisy UI nodes to simplify observations (e.g., combobox/menu/listbox internals).
        instruction = self.get_actor_instruction()
        last_action_is_optimal_navigate = False
        if len(self.previous_interactions["action"]) > 0:
            last_action = self.previous_interactions["action"][-1]
            if "optimal_navigate [" in last_action:
                last_action_is_optimal_navigate = True
        
        model_response_list = []
        action_element_list = [] # Keep all sampled actions for downstream Best-of-N/Judge selection.
        for _ in range(self.config.number): # Number of samples generated for the current step.
            get_valid_actions = False
            repetitive_note = False
            invalid_actions = False
            invalid_domain_retry_count = 0
            MAX_INVALID_DOMAIN_RETRIES = 3
            while not get_valid_actions:
                self.env_ref.obs = self.env_ref.webarena_env._get_obs()
                observation = self.env_ref.observation()
                url = self.env_ref.get_url()
                if "ups.com" in url:
                    self.env_ref.webarena_env.page.wait_for_timeout(5000)
                self.parent.update_online_state(url=url, observation=observation)
                self.update_online_state(url=url, observation=observation)
                online_input = self.get_online_input(criticism_elements=criticism_elements)
                if repetitive_note:
                    model_response = self.call_model_with_message(system_prompt=instruction+"\nGenerating the command `note [{}]` will be severely punished! Don't generate repetitive notes!".format(getattr(self, "note_buffer", "")), messages=self.arrange_message_for_model(online_input))
                elif last_action_is_optimal_navigate:
                    model_response = self.call_model_with_message(system_prompt=instruction+"\nThe goal of the optimal_navigate tool is to select the most suitable page, so repeatedly using this tool will cause the task to fail. Generating the command `optimal_navigate [{}]` will be severely punished! Don't generate repetitive optimal_navigate!".format(getattr(self, "optimal_navigate_buffer", "")), messages=self.arrange_message_for_model(online_input))
                elif invalid_actions:
                    model_response = self.call_model_with_message(system_prompt=instruction+"\nGenerating the command `{}` will be severely punished! Don't generate invalid actions! We don't have that element id in the current observation!".format(invalid_action_str), messages=self.arrange_message_for_model(online_input))
                else:
                    model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
                print(f"model_response: {model_response}")
                action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
                if "action" in action_elements.keys(): 
                    if self.are_valid_actions(action_elements["action"]):
                        note_buffer = getattr(self, "note_buffer", "")
                        if note_buffer and f"note [{note_buffer}" in action_elements["action"]:
                            print(f"Repetitive note: {note_buffer}")
                            repetitive_note = True
                            continue 
                        action_elements["input"] = online_input
                        if "optimal_navigate [" in  action_elements["action"]:
                            #  deloop mechanism for optimal_navigate
                            # if last_action_is_optimal_navigate == True and "optimal_navigate [" in model_response:
                            #     continue # regenerate model response
                            # parser opitmal_navigate action
                            match = re.search(r"optimal_navigate ?\[([\s\S]+?)\]\s*\[([\s\S]+?)\]\s*\[([\s\S]+?)\]", action_elements["action"])
                            think = match.group(1).strip()
                            if self.config.benchmark_name == "online_mind2web":
                                domain = self.required_sites[0]
                            else:
                                domain = match.group(2).strip()
                            query = match.group(3).strip()
                            
                            # Check if domain exists in domain_navigators
                            if domain not in self.domain_navigators:
                                print(f"[Error] Invalid domain '{domain}' not in domain_navigators. Available domains: {list(self.domain_navigators.keys())}")
                                invalid_domain_retry_count += 1
                                if invalid_domain_retry_count >= MAX_INVALID_DOMAIN_RETRIES:
                                    raise ValueError(f"Max retries ({MAX_INVALID_DOMAIN_RETRIES}) exceeded for invalid domain '{domain}'")
                                continue
                            
                            navigator = self.domain_navigators[domain]
                            if navigator is None:
                                print(f"Warning: domain navigator not found for {domain}")
                                invalid_domain_retry_count += 1
                                if invalid_domain_retry_count >= MAX_INVALID_DOMAIN_RETRIES:
                                    raise ValueError(f"Max retries ({MAX_INVALID_DOMAIN_RETRIES}) exceeded for None navigator '{domain}'")
                                continue
                            action_elements["navigate_think"] = think
                            action_elements["navigate_domain"] = domain
                            action_elements["navigate_query"] = query
                            if navigator.retriever.return_multivector:
                                topk_results = navigator.retriever.rerank(think + query)
                            else:
                                topk_results = navigator.retriever.search(think + query)
                                
                            action_elements["topk_webpages"] = topk_results
                            print(f"[Navigator] Retrieved {len(topk_results)} candidate pages")

                            # get image data
                            topk_images = []
                            for result in topk_results:
                                image = find_image_by_name(navigator.screenshots, result["image_name"])
                                if image:
                                    topk_images.append(image)
                                else:
                                    print(f"[Navigator] Warning: image not found: {result['image_name']}")
                            # construct multimodal messages
                            message_with_topk_images = self.webpage_selector_messages(
                                intent=self.objective,
                                query=think + query,
                                topk_images=topk_images
                            )
                            # llm reasoing select top-1 page
                            selection_response = self.call_page_selector_message(system_prompt=page_selector["instruction_template"], messages=message_with_topk_images)
                            print("selection_response:",selection_response)
                            selection_response = json_parser(selection_response)
                            if selection_response == "__JSON_PARSE_ERROR__":
                                continue
                            # If top-k has only one result, select it directly.
                            if len(topk_results) == 1:
                                print(f"[Navigator] topk=1, using the only result directly")
                                selection_response = {
                                    "target_page": topk_results[0]["image_name"]
                                }
                            print(f"[Navigator] Page selection response:\n{selection_response}")
                            # get target page and target webnode
                            action_elements["page_selection_response"] = selection_response
                            if selection_response["target_page"] == "None":
                                action_elements["target_page"] = None
                                print("[Navigator] Target page is None!!!")
                            else:
                                action_elements["target_page"] = selection_response["target_page"]
                                for id, webnode in navigator.webgraph.nodes.items():
                                    if webnode.screen_path is None:
                                        continue
                                    webnode_image_name = webnode.screen_path.split("/")[-1]
                                    if webnode_image_name == action_elements["target_page"]:
                                        action_elements["target_webnode"] = webnode
                                        break
                                action_elements["webgraph"] = navigator.webgraph
                                if "target_webnode" not in action_elements:
                                    continue
                        # --> end of optimal_navigate action
                        get_valid_actions = True
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    else:
                        invalid_action_str = action_elements["action"]
                        print(f"Invalid actions: {invalid_action_str}")
                        invalid_actions = True
                else:
                    raise NotImplementedError("You have to generate either action or action candidates.")
        # --> loop for num of actions
        self.verbose(instruction=instruction, online_input=online_input, model_response_list=model_response_list, action_element_list=action_element_list)

        if self.config.others.debug or self.config.debug:
            for i in range(len(action_element_list)):
                human_input = input(f"ACTION {i}: ")
                if human_input != "":
                    action_element_list[i]["action"] = human_input

        return action_element_list # dict_keys(['interaction history summary', 'observation description', 'reason', 'action', 'observation highlight', 'input'])

    def finalize_action(self, action_elements):
        self.get_observation_highlight(action_elements=action_elements)
        action = action_elements["action"]
        navigation_action = self.planning(action=action)
        if navigation_action:
            action_elements["navigation action"] = navigation_action
        action = self.take_note(action)
        action_elements["action"] = action
        return action_elements

class webNavigator:
    def __init__(self, 
                config = None, # Agent config loaded from the YAML file.
                prompt_dict: Dict = None,
                screenshot_config = None):
        
        self.config = config
        self.prompt_dict = {} if prompt_dict is None else prompt_dict
        self.objective = None
        self.online_observation = None
        self.online_url = None
        self.actor = None
        self.trajectory = []
        self.screenshot_config = screenshot_config
        self.screenshot_step_counter = 0
        self.task_id = None  # Will be set by eval.py
        self.log_dir = None  # Will be set for intermediate saving

    def init_actor(self):
        self.config.actor.others = self.config.others
        self.config.actor.benchmark_name = self.config.benchmark_name
        if len(self.sites) > 1:
            self.config.actor.navigation_command
        self.actor = Actor(
            config=self.config.actor,
            objective=self.objective,
            prompt_template=self.prompt_dict["actor"],
            plan_tree_node=PlanTreeNode(id=0, type="branch", text=f"Find the solution to \"{self.objective}\"", level=0, url=self.online_url, step=0),
            required_sites=self.sites
        )
        with open(self.actor.output_trash_path, "w") as _:
            pass

    def predict_action(self):
        action_element_list = self.actor.predict_action(criticism_elements={})
        selected_action_elements = self.actor.finalize_action(action_element_list[0])
        return { **selected_action_elements, **{}, **{}}, action_element_list


    def save_screenshot(self, observation, task_id):
        """Save screenshot from observation to file"""
        if not self.screenshot_config or not self.screenshot_config.get('enabled', False):
            return
            
        try:
            # Extract image data
            if isinstance(observation, dict) and 'image' in observation:
                image_data = observation['image']
                if image_data is not None:
                    # Convert numpy array to PIL Image
                    if isinstance(image_data, np.ndarray):
                        # Ensure the array is in the correct format (H, W, C)
                        if len(image_data.shape) == 3:
                            image = Image.fromarray(image_data.astype(np.uint8))
                            
                            # Create screenshot directory
                            screenshot_dir = os.path.join(self.screenshot_config['logdir'], 
                                                        self.screenshot_config['logname'], 
                                                        f"task_{task_id}")
                            os.makedirs(screenshot_dir, exist_ok=True)
                            
                            # Save screenshot
                            screenshot_path = os.path.join(screenshot_dir, f"step_{self.screenshot_step_counter}.png")
                            image.save(screenshot_path, "PNG")
                            print(f"[Screenshot] Saved step_{self.screenshot_step_counter}.png to {screenshot_dir}")
                            
                            self.screenshot_step_counter += 1
        except Exception as e:
            print(f"[Screenshot] Error saving screenshot: {e}")

    def save_unfinished_trajectory(self):
        """Save intermediate trajectory to a JSON file with _unfinished suffix.
        
        This allows monitoring task progress and debugging failed tasks.
        """
        if not self.log_dir or not self.task_id:
            return
        
        try:
            unfinished_file = os.path.join(self.log_dir, f"{self.task_id}_unfinished.json")
            log_data = {
                "task_id": self.task_id,
                "status": "unfinished",
                "num_steps": len(self.trajectory),
                "trajectory": self.trajectory,
            }
            with open(unfinished_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"[Unfinished] Saved intermediate trajectory to {unfinished_file} (step {len(self.trajectory)})")
        except Exception as e:
            print(f"[Unfinished] Error saving intermediate trajectory: {e}")

    def remove_unfinished_file(self):
        """Remove the unfinished trajectory file after task completion."""
        if not self.log_dir or not self.task_id:
            return
        
        try:
            unfinished_file = os.path.join(self.log_dir, f"{self.task_id}_unfinished.json")
            if os.path.exists(unfinished_file):
                os.remove(unfinished_file)
                print(f"[Cleanup] Removed {unfinished_file}")
        except Exception as e:
            print(f"[Cleanup] Error removing unfinished file: {e}")

    def act(self, objective, env):
        try:
            self.objective = objective
            self.source_objective = env.get_source_objective() if hasattr(env, 'get_source_objective') else objective
            self.sites = env.get_sites()
            observation = env.observation() 
            url = env.get_url()
            self.update_online_state(url=url, observation=observation)
            self.init_actor()
            self.actor.env_ref = env
            self.actor.parent = self
            
            # Save initial screenshot (step_0)
            if hasattr(self, 'task_id'):
                self.save_screenshot(observation, self.task_id)

            while not env.done():
                observation = env.observation()
                url = env.get_url()
                self.update_online_state(url=url, observation=observation)
                self.actor.update_online_state(url=url, observation=observation)
                action_elements, action_element_list = self.predict_action()
                action = action_elements["action"]
                navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")
                status = env.step(navigation_action, action_elements)
                
                # Save screenshot after each step
                if hasattr(self, 'task_id'):
                    new_observation = env.observation()
                    self.save_screenshot(new_observation, self.task_id)
                    
                if navigation_action and self.is_navigation(action=navigation_action) and status == False: # means invalid action
                    flaw_node = self.actor.active_node
                    flaw_node.note.append(f"STEP {self.get_step()}: You generate action \"{action}\", which has INVALID syntax. Strictly follow the action specifications.")          
                DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP = {
                    "observation": observation,
                    "action": action,
                    "url": url,
                    "plan": self.get_actor_active_plan(),
                    "reason": action_elements.get("reason", ""),
                    "observation highlight": action_elements.get("observation highlight", ""),
                    "retained element ids": action_elements.get("retained element ids", []),
                    "observation summary": action_elements.get("observation description", ""),
                    "mode analysis": action_elements.get("mode analysis", ""),
                    "navigate_think": action_elements.get("navigate_think", ""),
                    "navigate_domain": action_elements.get("navigate_domain", ""),
                    "navigate_query": action_elements.get("navigate_query", ""),
                    "page_selection_response": action_elements.get("page_selection_response", ""),
                    "topk_webpages": action_elements.get("topk_webpages", []),
                    "target_page": action_elements.get("target_page", None),
                    "target_webnode": action_elements.get("target_webnode", None),
                    "webgraph": action_elements.get("webgraph", None),
                }
                self.actor.update_history(**DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP)
                self.actor.del_observation_node()
                assert self.actor.equal_history_length()

                if len(action_element_list) > 1:
                    if self.config.others.logging:
                        self.log_step(
                            status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                            plan=self.get_actor_active_plan(),
                            **action_elements,
                            **{f"actor {i}:{k}": _action_elements[k] for i, _action_elements in enumerate(action_element_list) for k in _action_elements.keys() if k != "input" and k != "instruction"}
                        )
                        self.save_unfinished_trajectory()
                else:
                    if self.config.others.logging:
                        self.log_step(
                            status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                            plan=self.get_actor_active_plan(),
                            **action_elements,
                        )
                        self.save_unfinished_trajectory()

            # Task completed successfully, remove unfinished file
            self.remove_unfinished_file()
            return status if "status" in locals() and isinstance(status, dict) else env.status()
        
        except ValueError as e:
            # Catch invalid domain errors
            error_msg = str(e)
            if "Max retries" in error_msg and ("invalid domain" in error_msg or "None navigator" in error_msg):
                print(f"[Task Skip] {error_msg}")
                return {
                    'done': True,
                    'success': 0.0,
                    'reward': 0.0,
                    'num_actions': self.get_step(),
                    'error': 'invalid_domain_max_retries_exceeded'
                }
            else:
                raise


    def get_observation_text(self):
        if isinstance(self.online_observation, dict):
            return self.online_observation["text"]
        else:
            return self.online_observation

    def update_online_state(self, url, observation):
        self.online_url = url
        self.online_observation = observation

    def get_step(self):
        return self.actor.get_step()
    
    def is_navigation(self, action):
        return self.actor.is_navigation(action=action)
    
    def get_actor_active_plan(self):
        return self.actor.get_active_plan()
    
    def get_trajectory(self):
        return self.trajectory

    def log_step(self, status, **kwargs):
        def serialize_message_list(message_list):
            if not isinstance(message_list, list):
                return message_list
            return "".join([m[1] for m in message_list if m[0]=="text"])
        data_to_log = {}
        data_to_log['objective'] = getattr(self, 'source_objective', self.objective)
        data_to_log['url'] = self.online_url
        data_to_log['observation'] = self.get_observation_text()
        for (k, v) in status.items():
            data_to_log[k] = v
        for k in kwargs.keys():
            try:
                json.dumps(kwargs[k])
                data_to_log[k.replace(" ", "_")] = kwargs[k] if not "input" in k else serialize_message_list(kwargs[k])
            except:
                pass
        self.trajectory.append(data_to_log)