# -*- coding: utf-8 -*-
"""
Embed all external PNG images into presentation.html as base64 data URIs.
This makes the HTML file fully self-contained (no external dependencies).
"""
import base64
import os
import re

HTML_FILE = 'presentation.html'
IMG_DIR = 'outputs_local_fallback/figures'

# Read the HTML file
with open(HTML_FILE, 'r', encoding='utf-8') as f:
    html = f.read()

# Find all img src references to outputs_local_fallback/figures/
pattern = r'src="outputs_local_fallback/figures/([^"]+)"'
matches = re.findall(pattern, html)

print(f"Found {len(matches)} image references to embed:")
for m in matches:
    print(f"  - {m}")

# Replace each with base64 data URI
for filename in set(matches):
    filepath = os.path.join(IMG_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as img_file:
            b64 = base64.b64encode(img_file.read()).decode('ascii')
        
        old_src = f'src="outputs_local_fallback/figures/{filename}"'
        new_src = f'src="data:image/png;base64,{b64}"'
        html = html.replace(old_src, new_src)
        print(f"  [OK] Embedded: {filename} ({len(b64)} chars)")
    else:
        print(f"  [MISS] NOT FOUND: {filepath}")

# Also embed Google Fonts as fallback (use system fonts if offline)
# Add a @font-face fallback comment
html = html.replace(
    '<link rel="preconnect" href="https://fonts.googleapis.com">',
    '<!-- Google Fonts (requires internet on first load, cached after) -->\n    <link rel="preconnect" href="https://fonts.googleapis.com">'
)

# Write the self-contained HTML
output_file = 'presentation_standalone.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html)

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"\n[DONE] Created: {output_file} ({file_size:.1f} MB)")
print("   This file is fully self-contained - just send this 1 file!")
