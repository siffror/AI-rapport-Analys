import re

def extract_noterade_bolag_table(text: str) -> str:
    lines = text.splitlines()
    table_lines = []
    bolagsnamn_exempel = [
        "ABB", "Atlas", "Astra", "SEB", "Saab", "Nasdaq",
        "Epiroc", "Sobi", "Ericsson", "Wärtsilä", "Husqvarna", "Electrolux"
    ]
    for line in lines:
        if any(bolag in line for bolag in bolagsnamn_exempel) and re.search(r"\d", line):
            table_lines.append(line.strip())

    if not table_lines:
        return "⚠️ Ingen tabell med noterade bolag kunde identifieras."

    output = "Här är en tabell med noterade bolag:\n\n"
    output += "| Bolag | Övriga data |\n"
    output += "|-------|--------------|\n"
    for line in table_lines:
        parts = line.split()
        bolagsnamn = parts[0]
        rest = " ".join(parts[1:])
        output += f"| {bolagsnamn} | {rest} |\n"

    return output
