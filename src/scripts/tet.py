import pdfplumber

PDF_PATH = r"D:\College\legal_rag\legal_rag\data\acts\Transfer-of-Property.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        lines = text.split("\n")

        for line in lines:

            # print lines that look like section headers
            if "." in line and any(char.isdigit() for char in line):

                print(f"\nPAGE {page_num}")
                print(line)