from core.dla import run_dla
from core.ocr import run_ocr


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    # Step 1: Preprocessing
    # PDF 2 images
    # Deskew

    # Step 2: Document Layout Analysis
    # run_dla()

    # Step 3: OCR
    # If class in "text", "title", "table",...
    # run_ocr(extracts)
    # If class in "image", "figure", "logo", "stamp"
    # OCR not run

    pass
