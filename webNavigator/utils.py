import os
import base64
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_images(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load binary data for all PNG images in a folder.
    
    Args:
        folder_path: Image folder path.
        
    Returns:
        List[Dict]: Dict list containing image name and binary data.
        Format: [{"image_name": "xxx.png", "image_data": bytes}, ...]
        
    Raises:
        FileNotFoundError: Folder does not exist.
        PermissionError: No read permission.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"图片文件夹不存在: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"路径不是文件夹: {folder_path}")
    
    images = []
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    if not png_files:
        print(f"警告: 文件夹 {folder_path} 中没有找到PNG文件")
        return images
    
    print(f"加载图片文件夹: {folder_path} (找到 {len(png_files)} 个PNG文件)")
    
    for file in tqdm(png_files, desc="加载图片"):
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, "rb") as image_file:
                images.append({
                    "image_name": file, 
                    "image_data": image_file.read()
                })
        except Exception as e:
            print(f"警告: 无法读取图片 {file}: {e}")
            continue
    
    print(f"成功加载 {len(images)} 个图片文件")
    return images


def find_image_by_name(images: List[Dict[str, Any]], image_name: str) -> Optional[Dict[str, Any]]:
    """
    Find image data by image name.
    
    Args:
        images: Image data list.
        image_name: Target image name.
        
    Returns:
        Optional[Dict]: Matched image data; None if not found.
    """
    for image in images:
        if image["image_name"] == image_name:
            return image
    return None


def images_to_base64(images: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert image binary data to base64 encoding.
    
    Args:
        images: Image list containing binary data.
        
    Returns:
        List[Dict]: Image list containing base64-encoded data.
        Format: [{"image_name": "xxx.png", "image_data": "base64_string"}, ...]
    """
    base64_images = []
    print(f"转换 {len(images)} 个图片为base64格式")
    
    for image in tqdm(images, desc="转换base64"):
        try:
            image_data_base64 = base64.b64encode(image["image_data"]).decode('utf-8')
            base64_images.append({
                "image_name": image["image_name"], 
                "image_data": image_data_base64
            })
        except Exception as e:
            print(f"警告: 无法转换图片 {image['image_name']} 为base64: {e}")
            continue
    
    print(f"成功转换 {len(base64_images)} 个图片")
    return base64_images

def json_parser(text: str, raise_on_error: bool = True) -> Any:
    """
    Parse JSON content from text and handle markdown code fences.
    
    Args:
        text: Text that may contain JSON.
        raise_on_error: Whether to return an error marker on parse failure.
        
    Returns:
        Any: Parsed JSON result, or "__JSON_PARSE_ERROR__" on failure.
        
    Note:
        When raise_on_error=True and parsing fails, return "__JSON_PARSE_ERROR__"
        instead of raising an exception, so caller loops can continue.
    """
    import re
    
    if not text or not isinstance(text, str):
        if raise_on_error:
            raise ValueError(f"Invalid input type: {type(text)}, expected str")
        return text
    
    original_text = text  # Keep original text for error reporting.
    cleaned_text = text.strip()
    
    # Method 1: remove markdown code-fence markers (case-insensitive).
    # Match ```json / ```JSON / ``` prefixes.
    if re.match(r'^```[jJ][sS][oO][nN]?\s*\n?', cleaned_text):
        cleaned_text = re.sub(r'^```[jJ][sS][oO][nN]?\s*\n?', '', cleaned_text)
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    
    # Remove trailing code fence.
    if cleaned_text.rstrip().endswith("```"):
        cleaned_text = cleaned_text.rstrip()[:-3]
    
    cleaned_text = cleaned_text.strip()
    
    # Try direct JSON parsing first.
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: extract JSON object {...} using regex.
    json_object_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
    if json_object_match:
        try:
            return json.loads(json_object_match.group())
        except json.JSONDecodeError:
            pass
    
    # Method 3: use a looser regex for outermost {...}.
    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Method 4: try extracting JSON array [...].
    json_array_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    if json_array_match:
        try:
            return json.loads(json_array_match.group())
        except json.JSONDecodeError:
            pass
    
    # All parsing strategies failed.
    if raise_on_error:
        print(f"[Navigator-selector]JSON 解析失败，重新生成\n原始内容预览:\n{original_text}")
        # Return a marker instead of None so outer loops can continue.
        return "__JSON_PARSE_ERROR__"
    
    return text
