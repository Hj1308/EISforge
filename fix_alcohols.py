from pathlib import Path
import re
p = Path('app.py')
text = p.read_text(encoding='utf-8')
orig = text

# Force-fix main Alcohol list
text, n1 = re.subn(
    r'alcohol\s*=\s*st\.selectbox\("Alcohol",\s*\[[^\]]*\],\s*disabled=\(system_type\s*!?=\s*"AOR"\)\)',
    'alcohol = st.selectbox("Alcohol", ["ethanol", "methanol", "2-propanol", "ethylene glycol", "glycerol", "N/A"], disabled=(system_type != "AOR"))',
    text,
    count=1,
    flags=re.S,
)

# Force-fix alternate compact formatting if present
text = text.replace(
    'alcohol = st.selectbox("Alcohol",["ethanol","methanol","ethylene glycol","glycerol","N/A"], disabled=(system_type!="AOR"))',
    'alcohol = st.selectbox("Alcohol", ["ethanol", "methanol", "2-propanol", "ethylene glycol", "glycerol", "N/A"], disabled=(system_type != "AOR"))'
)

# Keep KL list consistent too
text = text.replace(
    'kl_alcohol = st.selectbox("Alcohol", ["ethanol","methanol","2-propanol","ethylene glycol","glycerol"], key="kl_alc")',
    'kl_alcohol = st.selectbox("Alcohol", ["ethanol", "methanol", "2-propanol", "ethylene glycol", "glycerol"], key="kl_alc")'
)

p.write_text(text, encoding='utf-8')
print('PATCHED app.py')
print('main has 2-propanol =', '2-propanol' in text and 'alcohol = st.selectbox("Alcohol"' in text)
print('changed =', text != orig)
