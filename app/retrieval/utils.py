import hashlib
import re


def file_sha256(path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def convert_section_headers(text):
    lines = text.splitlines()
    output = []

    for line in lines:
        match = re.match(
            r'^\*\*(\d+(?:\.\d+)*)\*\*\s+\*\*(.*?)\*\*\s*$',
            line.strip()
        )
        if match:
            number = match.group(1)
            title = match.group(2)
            level = number.count(".") + 1
            prefix = "#" * level
            output.append(f"{prefix} {number} {title}")
        else:
            output.append(line)

    return "\n".join(output)