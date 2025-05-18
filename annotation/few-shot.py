"""
few-shot.py
功能：调用gpt-4o api完成0-shot和5shot prompt下口语对应书面语及错误类型的标注
"""
import json
import time
import requests
import signal
import re
import os
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DS_API_KEY, OPENAI_API_KEY
from prompt import PROMPT_5SHOT, PROMPT_0SHOT

# 错误类型映射表（名称到数字）
ERROR_NAME_TO_NUM = {
    "句子成分缺失": 1,
    "句子结构混乱": 2,
    "句子成分错误": 3,
    "句子成分冗余": 4
}

interrupted = False

def handle_interrupt(signum, frame):
    global interrupted
    interrupted = True
    print("\n[警告] 捕获到 Ctrl+C，准备保存中断时的进度...")

signal.signal(signal.SIGINT, handle_interrupt)

def load_json(input_path):
    """加载JSON文件"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, output_path):
    """保存JSON文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_error_types(error_str):
    """解析错误类型字符串，返回数字数组"""
    if not error_str or error_str.strip() == "":
        return []
    error_str = error_str.replace("，", ",").replace("；", ",").replace("、", ",")
    error_numbers = []
    seen_numbers = set()
    for name, num in ERROR_NAME_TO_NUM.items():
        if name in error_str and num not in seen_numbers:
            error_numbers.append(num)
            seen_numbers.add(num)
    if not error_numbers:
        for part in error_str.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= 4:
                num = int(part)
                if num not in seen_numbers:
                    error_numbers.append(num)
                    seen_numbers.add(num)
    return sorted(error_numbers)

def process_data_item(item, prompt_template, max_retries=5):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            context = item["context"]  # 直接读取已有字段

            prompt = prompt_template.format(
                oral_sentence=item["spoken_text"],
                context=context
            )

            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result_text = response.json()["choices"][0]["message"]["content"]

            pattern = re.compile(
                r'翻译结果：\s*(.*?)\s*(?:错误类型：\s*(.*))?$',
                re.DOTALL
            )
            match = pattern.search(result_text)

            if match:
                translation = match.group(1).strip()
                error_str = match.group(2).strip() if match.group(2) else ""
            else:
                translation = "未识别到翻译结果"
                error_str = ""

            item["written_text"] = translation
            item["error_type"] = parse_error_types(error_str)

            return item  # 返回处理后的item

        except Exception as e:
            retry_count += 1
            print(f"[错误] API请求失败（ID: {item['id']}），重试 {retry_count}/{max_retries}：{e}")
            time.sleep(5)

    item["written_text"] = "API请求失败"
    item["error_type"] = []
    return item

from threading import Lock

from threading import Lock

def process_json_with_prompt(json_data, prompt_template, max_retries=5, progress_file=None):
    results = []
    processed_count = 0
    total_count = len(json_data)
    lock = Lock()

    def task_wrapper(item):
        result = process_data_item(item, prompt_template, max_retries)
        nonlocal processed_count
        with lock:
            processed_count += 1
            results.append(result)

            print(f"[进度] {processed_count}/{total_count} - ID: {item['id']}")
            print(f"  翻译结果：{result.get('written_text', '')}")
            print(f"  错误类型：{result.get('error_type', [])}")

            if progress_file and processed_count % 10 == 0:
                save_json(results, progress_file)
                print(f"[自动保存] 已保存前 {processed_count} 条到 {progress_file}\n")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(task_wrapper, item) for item in json_data]
        for future in as_completed(futures):
            future.result()

    if progress_file:
        save_json(results, progress_file)
        print(f"[最终保存] 已保存全部结果到 {progress_file}")

    return results



if __name__ == "__main__":
    # 处理数据 
    input_json = "dataset/test_dataset.json"
    start_row = 0
    num_rows = 2002

    try:
        all_data = load_json(input_json)
        data_slice = all_data[start_row:start_row + num_rows]

        for prompt_type, prompt_template in [("0shot", PROMPT_0SHOT)]: #("5shot", PROMPT_5SHOT)
            print(f"\n[开始] 使用 prompt 类型：{prompt_type}")
            output_file = f"dataset/output_test_{prompt_type}_gpt.json"
            backup_file = f"dataset/backup_test_{prompt_type}_gpt.json"

            input_copy = copy.deepcopy(data_slice)
            result = process_json_with_prompt(input_copy, prompt_template, progress_file=backup_file)
            save_json(result, output_file)
            print(f"[完成] {prompt_type} 结果已保存到 {output_file}")

    except Exception as e:
        print(f"\n[异常] 发生错误：{e}")
    finally:
        if interrupted:
            save_json(data_slice, "dataset/interrupted_backup_gpt.json")
            print(f"[中断] 已保存中断进度到 dataset/interrupted_backup_gpt.json")
    """
    # 重新处理失败项
    input_json = "dataset/test_dataset.json"
    output_json = "dataset/output_test_0shot_gpt.json"
    backup_json = "dataset/backup_test_0shot_gpt.json"

    failed_ids = [1211, 2686, 8168]

    try:
        # 1. 加载完整原始数据和原输出文件
        full_data = load_json(input_json)
        original_output = {item["id"]: item for item in load_json(output_json)} if os.path.exists(output_json) else {}

        # 2. 提取需要重跑的数据
        failed_items = [item for item in full_data if item["id"] in failed_ids]

        print(f"\n[开始] 重新处理失败项，共 {len(failed_items)} 条")

        # 3. 重跑失败项
        result = process_json_with_prompt(copy.deepcopy(failed_items), PROMPT_0SHOT, progress_file=backup_json)

        # 4. 合并新旧结果：替换原输出文件中对应 id 的项
        for item in result:
            original_output[item["id"]] = item

        # 5. 按 id 排序，转为 list 保存
        final_output = sorted(original_output.values(), key=lambda x: x["id"])

        save_json(final_output, output_json)
        print(f"[完成] 所有结果已合并并保存到 {output_json}")

    except Exception as e:
        print(f"\n[异常] 发生错误：{e}")
    finally:
        if interrupted:
            save_json(failed_items, "dataset/interrupted_backup_gpt.json")
            print(f"[中断] 已保存中断进度到 dataset/interrupted_backup_gpt.json")
    """


