"""
generate_context.py
功能：生成上下四句/全文+书面语上文/口语上文的context
"""
def generate_context(current_idx, data, context_mode=3, use_translated_context=True):
    """
    基于 JSON 数据生成上下文字符串。

    参数：
        current_idx (int): 当前处理项的索引
        data (List[Dict]): 所有数据项的 JSON 列表
        context_mode (int): 上下文模式（1=当前句，2=窗口1，3=窗口4，4=全文）
        use_translated_context (bool): 是否使用翻译后的前文（True）

    返回：
        str: 拼接后的上下文字符串
    """
    current_item = data[current_idx]
    current_file_id = current_item["file_id"]

    # 模式1：仅当前句子
    if context_mode == 1:
        return current_item["spoken_text"]

    # 模式2/3：窗口上下文（仅限相同文件序号）
    elif context_mode in [2, 3]:
        window_size = 1 if context_mode == 2 else 4

        # 向前查找（使用翻译内容，如果有）
        prev_context = []
        idx = current_idx - 1
        while len(prev_context) < window_size and idx >= 0:
            if data[idx]["file_id"] == current_file_id:
                text = data[idx].get("written_text") or data[idx]["spoken_text"]
                if text != "API请求失败":
                    prev_context.insert(0, text)
            idx -= 1

        # 向后查找（使用 spoken_text）
        next_context = []
        idx = current_idx + 1
        while len(next_context) < window_size and idx < len(data):
            if data[idx]["file_id"] == current_file_id:
                next_context.append(data[idx]["spoken_text"])
            idx += 1

        return "".join(prev_context + [current_item["spoken_text"]] + next_context)

    # 模式4：全文上下文（当前 file id 下的所有句子）
    elif context_mode == 4:
        full_context = []
        for idx, item in enumerate(data):
            if item["file_id"] == current_file_id:
                if use_translated_context and idx < current_idx and item.get("written_text") and item["written_text"] != "API请求失败":
                    full_context.append(item["written_text"])
                else:
                    full_context.append(item["spoken_text"])
        return "\n".join(full_context)

    else:
        raise ValueError(f"不支持的 context_mode: {context_mode}")