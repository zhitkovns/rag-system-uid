import re

input_file = "sources/DM2024_module9.txt"
output_file = "sources/DM2024_module9_fixed.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    # Удаляем служебные маркеры в строке
    line = re.sub(r'\s*\(\d+/\d+\)', '', line)
    line = re.sub(r'\s*\(продолжение\)', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\s*\(окончание\)', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\s*no_\d+\s*', ' ', line)
    line = re.sub(r'\s*implies\s*', ' ', line)
    # Нормализуем пробелы внутри строки
    line = re.sub(r'[ \t]+', ' ', line)
    # Удаляем пробелы в начале и конце
    line = line.strip()
    if line:  # не добавляем пустые строки
        cleaned_lines.append(line)

# Теперь объединяем строки обратно с переносами
text = '\n'.join(cleaned_lines)

# Дополнительно: переформулируем определения в формате "Термин — это ..."
def fix_definition(match):
    before = match.group(1).strip()
    after = match.group(2).strip()
    words = before.split()
    if len(words) > 1:
        term = words[-1]
        if term.endswith('ый') or term.endswith('ий') or term.endswith('ой'):
            term = term + ' граф'
        else:
            term = before
    else:
        term = before
    return f"{term} — это {after}."

text = re.sub(r'([А-ЯЁ][А-ЯЁа-яё\s]+?)\s+называется\s+([^.!?]+[.!?])', fix_definition, text)

# Заменяем двойные тире на обычные
text = text.replace('--', '—')

# Сохраняем
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print("Исправленный файл сохранён как", output_file)