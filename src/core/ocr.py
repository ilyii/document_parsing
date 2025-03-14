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
                    extracted.append({"words": word_text, "boxes": box})

    return extracted


def run_ocr(inp):
    predictor = ocr_predictor(
        "db_resnet50",
        pretrained=True,
        assume_straight_pages=False,
        preserve_aspect_ratio=True,
        # resolve_lines=True,
        # resolve_blocks=True,
    )
    res = predictor(inp)
    return res


def run_kie(inp):
    predictor = kie_predictor(
        "db_resnet50",
        pretrained=True,
        assume_straight_pages=False,
        preserve_aspect_ratio=True,
    )
    result = predictor(inp)
    predictions = result.pages[0].predictions
    for class_name in predictions.keys():
        list_predictions = predictions[class_name]
        for prediction in list_predictions:
            print(f"Prediction for {class_name}: {prediction}")


def visualize(res):
    # print(res.render())
    res.show()


if __name__ == "__main__":
    SRC = sys.argv[1]
    imagepaths = utils.find_images(SRC)
    # imagepaths = imagepaths[:1]
    docs = DocumentFile.from_images(imagepaths)
    res = run_ocr(docs)
    visualize(res)
    run_kie(docs)
