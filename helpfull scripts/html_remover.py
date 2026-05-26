#!/usr/bin/env python3
"""
Strip all HTML tags, CSS, and JavaScript from input.html and save plain text to remove_html.txt
Usage: python strip_html.py
"""

import re

def strip_html_css_js(text):
    # Remove <style>...</style> blocks (CSS)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove <script>...</script> blocks (JavaScript)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove inline style attributes
    text = re.sub(r'\s*style="[^"]*"', '', text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&nbsp;': ' ', '&quot;': '"', '&#39;': "'"
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)

    # Clean up excess whitespace/blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove empty lines
    return '\n'.join(lines)


if __name__ == '__main__':
    input_file = 'input.html'
    output_file = 'remove_html.txt'

    with open(input_file, 'r', encoding='utf-8') as f:
        raw = f.read()

    clean_text = strip_html_css_js(raw)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(clean_text)

    print(f"Done! Plain text saved to {output_file} ({len(clean_text)} characters)")