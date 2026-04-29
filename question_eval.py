import json
import re
import torch
from typing import Optional
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Konfiguration
# =========================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Modell & Tokenizer laden
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# =========================
# Tickettext
# =========================

post_text = """
Die Arbeitgeber schufen im August 22.000 neue Arbeitsplätze, wobei die Zahl der Beschäftigten im privaten Sektor um 38.000 stieg.\nDie Arbeitslosenquote stieg auf 4,3% (dies ist der höchste Stand seit 2021).\nDas Beschäftigungswachstum für Juni wurde nach unten korrigiert und weist nun einen Verlust von 13.000 Arbeitsplätzen aus, während das Wachstum für Juli von 73.000 auf 79.000 nach oben korrigiert wurde. Insgesamt wurden durch die Korrekturen 21.000 Arbeitsplätze gegenüber dem vorherigen Bericht abgezogen.\nDie Beschäftigungszahlen für Juni wurden von +14.000 auf -13.000 revidiert (erstmals negativ seit Dezember 2020).\nBei dem was eingepreist war, wären bessere Daten natürlich wünschenswerter gewesen. Nun wird die Marktmeinung (Zinssenkung September) bestätigt anstatt enttäuscht.
"""

# =========================
# Prompt (SLM-optimiert!)
# =========================

prompt = f"""
Vergib für den folgenden Text einen Wert zwischen 0 und 10. 

0 Bedeutet "Es handelt sich sicher um einer Frage".
10 bedeutet "Es handelt sich sicher um eine Antwort auf eine Frage."

\"\"\"
{post_text}
\"\"\"
"""

# =========================
# Inferenz
# =========================

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.0,
        do_sample=False
    )

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(" Rohantwort des Modells:\n", output_text)

# =========================
# Parsing & Validierung
# =========================

#try:
#    parsed = extract_json(output_text)
#    extraction = TicketExtraction(**parsed)

#    print("\n Validierte Extraktion:")
#    print(extraction.model_dump())

#except (ValueError, json.JSONDecodeError) as e:
#    print(" JSON-Parsing-Fehler:", e)

#except ValidationError as e:
#    print(" Schema-Validierungsfehler:", e)

