"""
annotation_multi.py
功能：调用deepseek-v3 api完成口语对应书面语及错误类型的标注
"""
import json
import time
import requests
import signal
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DS_API_KEY
from prompt import PROMPT_TEMPLATE
from generate_context import generate_context

interrupted = False

def handle_interrupt(signum, frame):
    global interrupted
    interrupted = True
    print("\n[警告] 捕获到 Ctrl+C，准备保存中断时的进度...")

signal.signal(signal.SIGINT, handle_interrupt)

def load_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_file_id_data(file_data, max_retries=5, progress_file=None):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DS_API_KEY}",
        "Content-Type": "application/json"
    }

    processed = 0
    total = len(file_data)

    for i in range(total):
        if interrupted:
            print("[中断] 捕获到 Ctrl+C，正在保存进度...")
            if progress_file:
                save_json(file_data, progress_file)
            break

        item = file_data[i]
        if item.get("written_text") and item["written_text"] != "API请求失败":
            continue  # 跳过已处理内容

        time.sleep(1)  # 控制速率

        retry_count = 0
        while retry_count < max_retries:
            try:
                context = generate_context(i, file_data)
                item["context"] = context

                prompt = PROMPT_TEMPLATE.format(
                    oral_sentence=item["spoken_text"],
                    context=context
                )

                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }

                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()

                result_text = response.json()["choices"][0]["message"]["content"]

                import re
                match = re.search(r'翻译结果：\s*\n?(.*?)(?:\n\n|\n$|$)', result_text, re.DOTALL)
                translation = match.group(1).strip() if match else "未识别到翻译结果"

                item["written_text"] = translation
                processed += 1
                print(f"进度：{processed}/{total} [ID: {item['id']}]")

                break

            except Exception as e:
                retry_count += 1
                print(f"[错误] API请求失败（ID: {item['id']}），重试 {retry_count}/{max_retries}：{e}")
                time.sleep(5)

        if retry_count == max_retries:
            item["written_text"] = "API请求失败"

    # 每处理完一个 file_id 的数据，就保存一次
    if progress_file:
        file_id = file_data[0]['file_id']  # 获取 file_id
        file_name = f"{progress_file}_{file_id}.json"  # 创建每个 file_id 独立的文件名
        save_json(file_data, file_name)
        print(f"[保存] 已保存当前 file_id 数据到 {file_name}")

    return file_data

def process_json(json_data, start_idx=0, max_retries=5, progress_file=None):
    # 按 file_id 分组数据
    grouped_data = {}
    for item in json_data[start_idx:]:
        file_id = item["file_id"]
        if file_id not in grouped_data:
            grouped_data[file_id] = []
        grouped_data[file_id].append(item)

    # 使用线程池来处理每个 file_id 分组的数据
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for file_id, file_data in grouped_data.items():
            futures.append(executor.submit(process_file_id_data, file_data, max_retries, progress_file))

        for future in as_completed(futures):
            result = future.result()
            all_results.extend(result)

    return all_results

if __name__ == "__main__":
    input_path = "dataset/input.json"
    output_path = "output_data_plus.json"
    backup_path = "interrupt_save_plus.json"
    start_row = 2
    num_rows = 10004

    try:
        data = load_json(input_path)
        result = process_json(data, start_row - 2, progress_file=backup_path)
        save_json(result, output_path)
        print(f"\n[完成] 数据已保存到 {output_path}")
    except Exception as e:
        print(f"\n[异常] 发生错误：{e}")
    finally:
        if interrupted:
            save_json(data, backup_path)
            print(f"\n[中断] 已保存进度到 {backup_path}")
