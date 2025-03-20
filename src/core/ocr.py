import sys

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import utils


def extract_words_and_boxes(json_data):
    extracted = []

    for page in json_data.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    word_text = word["value"]
                    box = [
                        int(word["geometry"][0][0] * page["dimensions"][0]),  # x1
                        int(word["geometry"][0][1] * page["dimensions"][1]),  # y1
                        int(word["geometry"][1][0] * page["dimensions"][0]),  # x2
                        int(word["geometry"][1][1] * page["dimensions"][1]),  # y2
                    ]
                    extracted.append({"text": word_text, "box": box})

    return extracted


def run_ocr(inp):
    predictor = ocr_predictor(
        "db_resnet50",
        pretrained=True,
        assume_straight_pages=False,
        preserve_aspect_ratio=True,
    )
    res = predictor(inp)
    extracted = extract_words_and_boxes(res)
    return extracted