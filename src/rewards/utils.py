import re


def extract_content(completion):
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return str(completion)
    if isinstance(completion, str):
        return completion
    return str(completion)


def extract_answer(text):
    content = extract_content(text)
    match = re.search(r"<answer>([\s\S]+?)</answer>", content)
    return match.group(1).strip() if match else content.strip()


def extract_thinking(text):
    content = extract_content(text)
    match = re.search(r"<think>([\s\S]+?)</think>", content)
    return match.group(1).strip() if match else ""
