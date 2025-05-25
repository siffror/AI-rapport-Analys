import easyocr
import os
import tempfile
from typing import Tuple

reader = easyocr.Reader(['sv', 'en'])

def extract_text_from_image_or_pdf(file) -> Tuple[str, str]:
    """
    Tar en PNG, JPG eller PDF, och returnerar OCR-utläst text.
    """
    suffix = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    result = reader.readtext(temp_path, detail=0)
    text = "\n".join(result)

    return text, temp_path
