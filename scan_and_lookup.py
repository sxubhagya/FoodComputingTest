import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import requests
from google.cloud import vision
from PIL import Image
from deep_translator import GoogleTranslator

def translate_to_english(text):
    if not text or text.strip() == "":
        return text
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # fallback to original if translation fails

def recognize_text_google_vision(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return [t.description for t in texts[1:]]
    else:
        return []

def recognize_web_entities(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    web_detection = response.web_detection
    if web_detection.web_entities:
        return [entity.description for entity in web_detection.web_entities if entity.description]
    else:
        return []

def recognize_image_logos(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    return [logo.description for logo in logos]

def recognize_image_labels(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return [label.description for label in labels]

def search_openfoodfacts_by_text(text):
    search_terms = text
    url = (
        "https://world.openfoodfacts.org/cgi/search.pl"
        f"?search_terms={search_terms}"
        "&search_simple=1&action=process&json=1"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        products = data.get("products", [])
        if products:
            product = products[0]
            return {
                "Product Name": translate_to_english(product.get("product_name", "Unknown")),
                "Brands": translate_to_english(product.get("brands", "Unknown")),
                "Categories": translate_to_english(product.get("categories", "Unknown")),
                "Ingredients": translate_to_english(product.get("ingredients_text", "Unknown")),
                "Allergens": product.get("allergens", "Unknown"),
            }
    return {"Error": "No products found for this search."}

def search_openfoodfacts_by_logo_and_label(logo, label):
    search_terms = f"{logo} {label}"
    return search_openfoodfacts_by_text(search_terms)

def detect_and_decode_barcode(image, image_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)

    # 1. Try barcode
    if barcodes:
        barcode_data = barcodes[0].data.decode("utf-8")
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_data}.json"
        response = requests.get(url)
        if response.status_code == 200:
            product = response.json().get("product", {})
            return {
                "Product Name": translate_to_english(product.get("product_name", "Unknown")),
                "Brands": translate_to_english(product.get("brands", "Unknown")),
                "Categories": translate_to_english(product.get("categories", "Unknown")),
                "Ingredients": translate_to_english(product.get("ingredients_text", "Unknown")),
                "Energy (kcal/100g)": product.get("nutriments", {}).get("energy-kcal_100g", "Unknown"),
                "Fat (g/100g)": product.get("nutriments", {}).get("fat_100g", "Unknown"),
                "Sugars (g/100g)": product.get("nutriments", {}).get("sugars_100g", "Unknown"),
                "Allergens": product.get("allergens", "Unknown"),
            }

    # 2. Try Google Vision OCR
    texts = recognize_text_google_vision(image_path)
    if texts:
        result = search_openfoodfacts_by_text(" ".join(texts[:3]))
        if "Error" not in result:
            return result

    # 3. Try Google Vision Web Detection
    web_entities = recognize_web_entities(image_path)
    if web_entities:
        result = search_openfoodfacts_by_text(" ".join(web_entities[:2]))
        if "Error" not in result:
            return result

    # 4. Try logo and label
    labels = recognize_image_labels(image_path)
    logos = recognize_image_logos(image_path)
    if logos and labels:
        result = search_openfoodfacts_by_logo_and_label(logos[0], labels[0])
        if "Error" not in result:
            return result
    elif labels:
        result = search_openfoodfacts_by_logo_and_label("", labels[0])
        if "Error" not in result:
            return result
    elif logos:
        result = search_openfoodfacts_by_logo_and_label(logos[0], "")
        if "Error" not in result:
            return result

    return {"Error": "No product found by any method."}

# The following lines are for CLI/testing only. Remove or comment out for Streamlit use.
if __name__ == "__main__":
    image_path = "/Users/soubhagya/Desktop/barcodespy/oreo.png"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or path is incorrect.")
    result = detect_and_decode_barcode(image, image_path)
    for k, v in result.items():
        print(f"{k}: {v}")