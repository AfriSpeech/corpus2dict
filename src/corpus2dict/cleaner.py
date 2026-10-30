"""Cleaner module for stripping code blocks, structured syntax, and programming identifiers."""

import re

# Comprehensive set of programming reserved keywords across major languages
CODE_KEYWORDS = {
    "function", "functions", "return", "returns", "def", "class", "classes", "import", "imports",
    "const", "var", "let", "args", "kwargs", "self", "lambda", "void", "null", "undefined",
    "true", "false", "boolean", "bool", "int", "integer", "float", "char", "string", "strings",
    "public", "private", "protected", "static", "package", "namespace", "struct", "enum",
    "typedef", "sizeof", "typeof", "instanceof", "stdout", "stdin", "stderr", "printf", "println",
    "console", "argv", "argc", "elif", "endif", "ifndef", "goto", "yield", "async", "await",
    "tuple", "tuples", "dict", "dicts", "array", "arrays", "params", "parameters", "nil"
}

CODE_CHARS = set("{}[]();/\\*+=<>~#$@|&^%!`")

# Code file extensions that should be rejected if appearing as a token
CODE_EXTENSIONS_PATTERN = re.compile(
    r"\.(py|js|ts|jsx|tsx|cpp|c|h|hpp|java|go|rb|rs|php|html|css|json|yaml|yml|sh|md|txt|csv|tar|gz|zip|exe|bin|dll|so)$",
    re.IGNORECASE
)

# Common CLI/shell commands and code declarations starting a line
CODE_LINE_STARTERS = (
    "import ", "from ", "def ", "class ", "return ", "export ", "const ", "let ", "var ",
    "git ", "pip ", "npm ", "pnpm ", "yarn ", "docker ", "curl ", "wget ", "sudo ", "cd ",
    "mkdir ", "chmod ", "chown ", "ls ", "grep ", "bash ", "sh ", "cat ", "#include", "#define",
    "package ", "use strict", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "FROM ", "WHERE "
)

def strip_structured_text(text: str) -> str:
    """Strip code blocks, inline code, math, URLs, markup, and CLI lines from text.
    
    Returns clean natural language prose ready for linguistic parsing.
    """
    if not text:
        return ""
    # 1. Multi-line code blocks (```...```)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # 2. LaTeX display math ($$...$$, \[...\])
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    # 3. Inline code (`...`)
    text = re.sub(r"`[^`]+`", " ", text)
    # 4. Inline math ($...$, \(...\))
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"\\\([^\)]+\\\)", " ", text)
    # 5. HTML / XML tags (<...>)
    text = re.sub(r"<[^>]+>", " ", text)
    # 6. URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # 7. Emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    # 8. File paths & system paths
    text = re.sub(r"(?:/[a-zA-Z0-9_\-\.]+)+|(?:[a-zA-Z]:\\[a-zA-Z0-9_\-\.\\]+)", " ", text)
    
    # 9. Clean line by line (skip tables, CLI commands, code declarations)
    lines = []
    for line in text.split("\n"):
        sline = line.strip()
        if not sline:
            continue
        # Skip markdown table rows
        if sline.startswith("|") and sline.endswith("|"):
            continue
        # Skip lines starting with code statements or CLI commands
        if sline.startswith(CODE_LINE_STARTERS):
            continue
        # Skip lines containing only syntax brackets and operators
        if re.match(r"^[{}();,\[\]=><:\-\+*/\\|&^%#@~`\'\"]+$", sline):
            continue
        lines.append(sline)
        
    cleaned = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return cleaned

def is_code_token(text: str) -> bool:
    """Check if a token represents code syntax, identifiers, or variables."""
    # snake_case (e.g., user_id, api_key)
    if "_" in text:
        return True
    # Syntax punctuation characters
    if any(c in CODE_CHARS for c in text):
        return True
    # File extensions
    if CODE_EXTENSIONS_PATTERN.search(text):
        return True
    # camelCase or PascalCase with lowercase-to-uppercase transitions (e.g., userId)
    if re.search(r"[a-z][A-Z]", text):
        return True
    # Hex or pure digits
    if re.match(r"^0x[0-9a-fA-F]+$", text) or text.isdigit():
        return True
    return False

def chunk_long_text(text: str, max_chars: int = 50000) -> list[str]:
    """Split exceptionally long text (e.g. pasted logs/dumps) to avoid parser memory spikes."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    current = []
    curr_len = 0
    for p in text.split("\n"):
        if curr_len + len(p) > max_chars and current:
            parts.append(" ".join(current))
            current = [p]
            curr_len = len(p)
        else:
            current.append(p)
            curr_len += len(p) + 1
    if current:
        parts.append(" ".join(current))
    final_parts = []
    for pt in parts:
        if len(pt) <= max_chars:
            final_parts.append(pt)
        else:
            for i in range(0, len(pt), max_chars):
                chunk = pt[i:i+max_chars].strip()
                if len(chunk) > 10:
                    final_parts.append(chunk)
    return final_parts
