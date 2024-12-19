import re
import string

target_ignore_regex = re.compile(r"(?s).*#### ")
output_ignore_regex = re.compile(r"(?s).*The answer is ")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def process_output_top1(output: str) -> str:
    output = output_ignore_regex.sub("", output)
    output = normalize_text(output)
    return output

def process_target(output: str) -> str:
    output = target_ignore_regex.sub("", output)
    output = normalize_text(output)
    return output