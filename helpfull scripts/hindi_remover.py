import re

input_file = "input_hindi.txt"
output_file = "output_hindi.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Remove bracketed Hindi text:
# (हिन्दी), [हिन्दी], {हिन्दी}
text = re.sub(r'\(\s*[\u0900-\u097F\s]+\s*\)', '', text)
text = re.sub(r'\[\s*[\u0900-\u097F\s]+\s*\]', '', text)
text = re.sub(r'\{\s*[\u0900-\u097F\s]+\s*\}', '', text)

# Remove Hindi text that follows separators:
# , हिन्दी
# - हिन्दी
# : हिन्दी
# ; हिन्दी
text = re.sub(
    r'\s*[,;:\-–—]\s*[\u0900-\u097F\s]+',
    '',
    text
)

# Remove remaining Hindi characters
text = re.sub(r'[\u0900-\u097F]+', '', text)

# Remove empty brackets left behind
text = re.sub(r'\(\s*\)', '', text)
text = re.sub(r'\[\s*\]', '', text)
text = re.sub(r'\{\s*\}', '', text)

# Normalize spaces
text = re.sub(r'[ \t]+', ' ', text)

# Remove spaces before punctuation
text = re.sub(r'\s+([,.;:!?])', r'\1', text)

# Collapse multiple blank lines
text = re.sub(r'\n\s*\n+', '\n\n', text)

# Strip trailing spaces on each line
text = '\n'.join(line.strip() for line in text.splitlines())

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved cleaned text to {output_file}")