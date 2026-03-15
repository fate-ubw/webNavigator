import os
import time
import re
import argparse
import shutil
from dotenv import load_dotenv
load_dotenv()

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from webNavigator.env import WebArenaEnvironmentWrapper

from webNavigator.data_prep import *
from webNavigator.prompts import webnavigator_prompt
from webNavigator import webNavigator

def run():
    parser = argparse.ArgumentParser(description="Only the config file argument should be passed")
    parser.add_argument("--config", type=str, required=True, help="yaml config file location")
    parser.add_argument("--shard-index", type=int, default=0, help="Current worker index (starting from 0)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (parallel worker count)")
    parser.add_argument("--use-virtual-display", action="store_true", help="Use virtual display (requires Xvfb and pyvirtualdisplay)")
    args = parser.parse_args()
    display = None
    if args.use_virtual_display:
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1920, 1080))
        display.start()
    
    try:
        
        with open(args.config, "r") as file:
            config = DotDict(yaml.safe_load(file))
        
        if config.logging:
            if hasattr(config.agent, "actor") and hasattr(config.agent.actor, "model"):
                model_name = config.agent.actor.model.split("/")[-1]
            elif hasattr(config.agent, "model_name"):
                model_name = config.agent.model_name.split("/")[-1]
            else:
                model_name = ""
            
            if config.logname:
                base_logname = config.logname
                if model_name and model_name not in base_logname:
                    dstdir = f"{config.logdir}/{base_logname}-{model_name}"
                else:
                    dstdir = f"{config.logdir}/{base_logname}"
            else:
                if model_name:
                    dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}-{model_name}"
                else:
                    dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(dstdir, exist_ok=True)
            shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
        random.seed(42)
        
        config_file_list = []
        if config.benchmark is None or config.benchmark.name is None:
            raise ValueError("benchmark or benchmark.name is not set")
        task_name = config.benchmark.task_name
        if task_name is None:
            raise ValueError("task_name is not set")
        task_ids = config.env.task_ids
        if task_ids == "all" or task_ids == ["all"]:
            task_ids = [filename[:-len(".json")] for filename in os.listdir(f"config_files/{config.benchmark.name}/{task_name}") if filename.endswith(".json")]
            task_ids.sort(key=lambda x: int(x) if x.isdigit() else x)  
        
        if args.num_shards > 1:
            task_ids = task_ids[args.shard_index::args.num_shards]
            print(f"[Shard {args.shard_index}/{args.num_shards}] 处理 {len(task_ids)} 个任务: {task_ids}")
            random.seed(int(time.time() * 1000) + args.shard_index)
            random.shuffle(task_ids)
            print(f"[Shard {args.shard_index}/{args.num_shards}] 任务顺序打乱后: {task_ids}")
            random.seed(42)

        for task_id in task_ids:
            config_file_list.append(f"config_files/{config.benchmark.name}/{task_name}/{task_id}.json")
        fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
        current_viewport_only = not fullpage 

        config.agent.benchmark_name = config.benchmark.name

        if config.agent.type == "webNavigator":
            # Prepare screenshot config
            screenshot_config = None
            if hasattr(config, 'screenshot_storage') and config.screenshot_storage.enabled:
                screenshot_config = {
                    'enabled': config.screenshot_storage.enabled,
                    'full_page': config.screenshot_storage.full_page,
                    'logdir': dstdir.rsplit('/', 1)[0],  # Parent directory of dstdir
                    'logname': dstdir.rsplit('/', 1)[1]   # Last part of dstdir
                }
            
            agent_init = lambda: webNavigator(
                config = config.agent,
                prompt_dict = {k: v for k, v in webnavigator_prompt.__dict__.items() if isinstance(v, dict)},
                screenshot_config = screenshot_config,
            )
        else:
            raise NotImplementedError(f"{config.agent.type} not implemented")

        for config_file in config_file_list: # 
            with open(config_file, "r") as f:
                task_config = json.load(f)
                print(f"Task {task_config['task_id']}.")
            if os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
                print(f"Skip {task_config['task_id']}.")
                continue
            
            # Clean up old screenshot directory if exists
            task_screenshot_dir = os.path.join(dstdir, f"task_{task_config['task_id']}")
            if os.path.exists(task_screenshot_dir):
                # Count PNG images before deletion
                image_files = [f for f in os.listdir(task_screenshot_dir) if f.endswith('.png')]
                num_images = len(image_files)
                # Print cleanup information
                print(f"[Cleanup] Task {task_config['task_id']}:")
                print(f"  └─ Removing task_{task_config['task_id']}/ ({num_images} images)")
                # Remove the entire directory
                shutil.rmtree(task_screenshot_dir)
            
            if task_config['task_id'] in list(range(600, 650))+list(range(681, 689)):
                print("Reddit post task. Sleep 2 mins.")
                time.sleep(120)
            # Try to initialize env with 3 retries
            env = None
            init_success = False
            for retry in range(3):
                try:
                    env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                                    max_browser_rows=config.env.max_browser_rows, 
                                                    max_steps=config.max_steps, 
                                                    slow_mo=1, 
                                                    observation_type="accessibility_tree", 
                                                    current_viewport_only=current_viewport_only, 
                                                    viewport_size={"width": 1920, "height": 1080}, 
                                                    headless=config.env.headless,
                                                    global_config=config)
                    init_success = True
                    break
                except PlaywrightTimeoutError as e:
                    print(f"[Env Init Timeout] Attempt {retry+1}/3 for task {task_config['task_id']}: {e}")
                    if env and hasattr(env, 'webarena_env'):
                        try:
                            env.webarena_env.close()
                        except:
                            pass
                    if retry == 2:
                        print(f"[Task {task_config['task_id']} Skip] Max retries exceeded")
                        break
                    time.sleep(5)
            
            if not init_success:
                continue
            
            agent = agent_init()
            if hasattr(agent, 'screenshot_config') and agent.screenshot_config:
                agent.task_id = task_config['task_id']
            # Set log_dir for intermediate trajectory saving
            if hasattr(agent, 'log_dir'):
                agent.log_dir = dstdir
                agent.task_id = task_config['task_id']
                
            objective = env.get_objective()
            status = agent.act(objective=objective, env=env) # {'done': True, 'reward': 0.0, 'success': 0.0, 'num_actions': 1}
            env.close()
            
            # Check if task failed with error - skip saving if error exists
            if isinstance(status, dict) and status.get('error'):
                print(f"[Task {task_config['task_id']} Failed] Error: {status['error']}")
                print(f"Task {task_config['task_id']} will not be saved and can be re-run later.")
                continue
            
            if config.logging:
                with open(config_file, "r") as f:
                    task_config = json.load(f)
                log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
                log_data = {
                    "task": config_file,
                    "id": task_config['task_id'],
                    "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                    "type": config.agent.type,
                    "trajectory": agent.get_trajectory(),
                }
                summary_file = os.path.join(dstdir, "summary.csv")
                summary_data = {
                    "task": config_file,
                    "task_id": task_config['task_id'],
                    "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                    "type": config.agent.type,
                    "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
                }
                if status:
                    summary_data.update(status)
                log_run(
                    log_file=log_file,
                    log_data=log_data,
                    summary_file=summary_file,
                    summary_data=summary_data,
                )
    except:
        import traceback
        traceback.print_exc()
    finally:
        if display is not None:
            display.stop()


if __name__ == "__main__":
    run()
