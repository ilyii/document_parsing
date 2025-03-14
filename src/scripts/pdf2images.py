import os
import sys
import fitz
from PIL import Image

SRC = sys.argv[1]
OUTDIR = sys.argv[2]

if __name__ == "__main__":
    if not os.path.exists(SRC):
        exit("File not found")

    if os.path.isfile(SRC):
        pdfs = [SRC]
    
    elif os.path.isdir(SRC):
        pdfs = [os.path.join(SRC, f) for f in os.listdir(SRC) if f.endswith(".pdf")]

    else:
        exit()

    os.makedirs(OUTDIR, exist_ok=True)
        
    for idx, pdf in enumerate(pdfs):
        doc = fitz.open(pdf)
        for pidx, page in enumerate(doc):
            pix = page.get_pixmap(matrix = fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(os.path.join(OUTDIR, f"{os.path.basename(pdf).replace('.pdf', '')}_{pidx:02}.png"))

