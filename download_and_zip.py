"""
download_and_zip.py — Download Open Images locally, structure like your dataset, zip for upload
================================================================================================
Run locally:
    pip install fiftyone pillow tqdm pyarrow pandas
    python download_and_zip.py

Output:
    - external_images/images/{category_id}/*.jpg  (same structure as your train/images/)
    - external_images/external_train.parquet       (same schema as train.parquet)
    - external_images.rar                          (ready to upload to Drive)
"""

import os
import uuid
import json
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\karim\Desktop\Data Science Projects\Vast AI\Data"
OUTPUT_DIR    = r"C:\Users\karim\Desktop\Data Science Projects\Vast AI\Data\external_images"
IMAGE_DIR     = os.path.join(OUTPUT_DIR, "images")
TAXONOMY_PATH = os.path.join(BASE_DIR, "level2_categories.json")
os.makedirs(IMAGE_DIR, exist_ok=True)

MAX_IMAGES_PER_CLASS = 150  # target ~150 per class

# ── Category to Open Images label mapping ─────────────────────────────────────
CATEGORY_TO_OI_LABELS = {
    1:  ["Animal"],
    2:  ["Pet supply"],
    3:  ["Clothing", "Dress", "Shirt", "Pants", "Jacket"],
    4:  ["Fashion accessory", "Hat", "Scarf", "Belt"],
    5:  ["Costume"],
    6:  ["Handbag"],
    7:  ["Handbag", "Wallet"],
    8:  ["Jewellery", "Necklace", "Ring", "Earrings"],
    9:  ["Shoe"],
    10: ["Footwear", "Boot", "Sneakers", "High heels"],
    11: ["Ticket"],
    12: ["Art", "Craft"],
    13: ["Party supply"],
    14: ["Baby bathing"],
    15: ["Baby gift"],
    16: ["Baby health"],
    17: ["Baby safety"],
    18: ["Baby toy", "Toy"],
    19: ["Stroller"],
    20: ["Baby transport accessory"],
    21: ["Diaper"],
    22: ["Baby bottle"],
    23: ["Potty"],
    24: ["Baby blanket"],
    25: ["Advertising"],
    26: ["Agricultural equipment", "Tractor"],
    27: ["Industrial equipment"],
    28: ["Construction equipment", "Crane"],
    29: ["Dental equipment"],
    30: ["Camera", "Film equipment"],
    31: ["Finance"],
    32: ["Food service equipment"],
    33: ["Chainsaw"],
    34: ["Hair dryer"],
    35: ["Heavy machinery", "Excavator"],
    36: ["Hotel"],
    37: ["Shelf"],
    38: ["Industrial storage accessory"],
    39: ["Janitorial cart"],
    40: ["Police equipment"],
    41: ["Manufacturing equipment"],
    42: ["Forklift"],
    43: ["Medical equipment"],
    45: ["Tattoo equipment"],
    46: ["Retail equipment"],
    47: ["Laboratory equipment", "Microscope"],
    48: ["Sign"],
    49: ["Safety equipment", "Helmet"],
    50: ["Camera accessory", "Tripod"],
    51: ["Camera", "Digital camera"],
    52: ["Binoculars", "Telescope"],
    53: ["Photography equipment"],
    54: ["Arcade game"],
    55: ["Headphones", "Speaker", "Microphone"],
    56: ["Electronic component", "Circuit board"],
    57: ["Telephone", "Mobile phone"],
    58: ["Electronic component"],
    59: ["Computer", "Laptop"],
    60: ["Computer accessory", "Keyboard", "Mouse"],
    61: ["GPS"],
    62: ["GPS navigation"],
    63: ["GPS tracker"],
    64: ["Marine electronics"],
    65: ["Router"],
    66: ["Printer", "Scanner"],
    67: ["Radar detector"],
    68: ["Speed radar"],
    69: ["Toll device"],
    70: ["Television", "Monitor"],
    71: ["Video game accessory", "Joystick"],
    72: ["Video game console"],
    73: ["Drink", "Juice", "Beer", "Wine"],
    74: ["Food", "Fruit", "Vegetable"],
    75: ["Tobacco", "Cigarette"],
    76: ["Crib"],
    77: ["Bed", "Mattress"],
    78: ["Bench"],
    79: ["Cabinet", "Wardrobe"],
    80: ["Cart"],
    81: ["Chair accessory"],
    82: ["Chair", "Armchair"],
    83: ["Television"],
    84: ["Furniture set"],
    85: ["Futon"],
    86: ["Futon pad"],
    87: ["Futon"],
    88: ["Desk"],
    89: ["Office furniture accessory"],
    90: ["Ottoman"],
    91: ["Garden furniture"],
    92: ["Outdoor furniture accessory"],
    93: ["Room divider accessory"],
    94: ["Room divider"],
    95: ["Shelf", "Bookcase"],
    96: ["Shelving accessory"],
    97: ["Sofa accessory"],
    98: ["Sofa", "Couch"],
    99: ["Table accessory"],
    100: ["Table", "Dining table", "Coffee table"],
    101: ["Building material"],
    102: ["Wood", "Brick"],
    103: ["Fence", "Gate"],
    104: ["Fuel"],
    105: ["Gas can"],
    106: ["Hardware accessory"],
    107: ["Pump"],
    108: ["Air conditioner", "Heater"],
    109: ["Lock", "Key"],
    110: ["Pipe", "Faucet"],
    111: ["Electrical supply", "Wire"],
    112: ["Engine"],
    113: ["Storage tank"],
    114: ["Tool accessory"],
    115: ["Hammer", "Screwdriver", "Wrench", "Tool"],
    116: ["Medical equipment"],
    117: ["Jewellery"],
    118: ["Cosmetics", "Toothbrush"],
    119: ["Towel", "Soap dispenser"],
    120: ["Security camera", "Alarm"],
    121: ["Vase", "Candle"],
    122: ["Fire extinguisher"],
    123: ["Fireplace accessory"],
    124: ["Fireplace"],
    125: ["Smoke detector"],
    126: ["Appliance accessory"],
    127: ["Washing machine", "Refrigerator"],
    128: ["Cleaning supply"],
    129: ["Kitchen utensil", "Cookware", "Cutlery"],
    130: ["Garden tool", "Lawn mower"],
    131: ["Lamp", "Chandelier"],
    132: ["Lighting accessory"],
    133: ["Pillow", "Blanket"],
    134: ["Umbrella", "Parasol"],
    135: ["Plant", "Flower"],
    136: ["Swimming pool"],
    137: ["Ashtray"],
    138: ["Umbrella"],
    139: ["Wood stove"],
    140: ["Backpack"],
    141: ["Briefcase"],
    142: ["Cosmetic bag"],
    143: ["Diaper bag"],
    144: ["Dry box"],
    145: ["Duffel bag"],
    146: ["Fanny pack"],
    147: ["Garment bag"],
    148: ["Luggage accessory"],
    149: ["Messenger bag"],
    150: ["Tote bag"],
    151: ["Suitcase", "Luggage"],
    152: ["Train case"],
    153: ["Erotic"],
    154: ["Weapon", "Gun", "Knife"],
    155: ["Book"],
    156: ["Woodworking"],
    157: ["DVD"],
    158: ["Magazine", "Newspaper"],
    159: ["Musical instrument", "Vinyl"],
    160: ["Manual"],
    161: ["Sheet music"],
    162: ["Bookmark"],
    163: ["Desk pad"],
    164: ["Binder"],
    165: ["Pen", "Stapler"],
    166: ["Impulse sealer"],
    167: ["Lap desk"],
    168: ["Name plate"],
    169: ["Chair mat"],
    170: ["Office cart"],
    171: ["Shredder"],
    172: ["Calculator"],
    173: ["Paper", "Envelope"],
    174: ["Whiteboard"],
    175: ["Box", "Tape"],
    176: ["Memorial supply"],
    177: ["Cross", "Menorah"],
    178: ["Wedding cake"],
    179: ["Computer software"],
    180: ["Digital goods"],
    181: ["Video game"],
    182: ["Ball", "Sports equipment"],
    183: ["Dumbbell", "Treadmill"],
    184: ["Billiards", "Table tennis"],
    185: ["Tent", "Kayak", "Bicycle"],
    186: ["Game timer"],
    187: ["Board game"],
    188: ["Swing", "Slide"],
    189: ["Puzzle"],
    190: ["Toy", "Doll", "Action figure"],
    191: ["Auto part", "Car part"],
    192: ["Car", "Motorcycle", "Truck"],
}

# ── Install fiftyone if needed ─────────────────────────────────────────────────
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    print("FiftyOne ready")
except ImportError:
    print("Installing fiftyone...")
    os.system("pip install fiftyone")
    import fiftyone as fo
    import fiftyone.zoo as foz

# ── Download ──────────────────────────────────────────────────────────────────
records     = []
failed_cats = []

print(f"\nDownloading up to {MAX_IMAGES_PER_CLASS} images per category")
print(f"Output: {IMAGE_DIR}\n")

for category_id, oi_labels in tqdm(CATEGORY_TO_OI_LABELS.items()):
    cat_dir = os.path.join(IMAGE_DIR, str(category_id))
    os.makedirs(cat_dir, exist_ok=True)

    # Skip if already downloaded enough
    existing = len([f for f in os.listdir(cat_dir) if f.endswith(".jpg")])
    if existing >= MAX_IMAGES_PER_CLASS:
        print(f"  cat {category_id}: already has {existing} images, skipping")
        continue

    try:
        dataset_name = f"oi_cat_{category_id}"
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name)

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["classifications"],
            classes=oi_labels,
            max_samples=MAX_IMAGES_PER_CLASS,
            seed=42,
            shuffle=True,
            dataset_name=dataset_name,
        )

        downloaded = 0
        for sample in dataset:
            try:
                anon_id   = str(uuid.uuid4())
                dest_path = os.path.join(cat_dir, f"{anon_id}.jpg")
                img = Image.open(sample.filepath).convert("RGB")
                img = img.resize((300, 300), Image.LANCZOS)
                img.save(dest_path, "JPEG", quality=95)

                records.append({
                    "anonymous_id": anon_id,
                    "category_id":  category_id,
                    "level_2_id":   category_id,
                    "color_label":  "unknown",
                    "gender":       None,
                    "color":        None,
                    "source":       "open_images"
                })
                downloaded += 1

            except Exception:
                continue

        print(f"  cat {category_id}: {downloaded} images")
        fo.delete_dataset(dataset_name)

    except Exception as e:
        print(f"  cat {category_id}: FAILED — {e}")
        failed_cats.append(category_id)
        continue

# ── Save parquet ──────────────────────────────────────────────────────────────
if records:
    external_df  = pd.DataFrame(records)
    parquet_path = os.path.join(OUTPUT_DIR, "external_train.parquet")
    external_df.to_parquet(parquet_path, index=False)
    print(f"\nSaved {len(external_df)} records to {parquet_path}")
    print(f"Categories covered: {external_df['category_id'].nunique()}")
    print(f"Failed: {len(failed_cats)} — {failed_cats}")

# ── Zip the output folder ─────────────────────────────────────────────────────
print("\nZipping external_images folder...")
zip_path = os.path.join(BASE_DIR, "external_images")
shutil.make_archive(zip_path, "zip", BASE_DIR, "external_images")
print(f"Saved zip to: {zip_path}.zip")
print("\nDone! Upload external_images.zip to Google Drive then on Vast.ai:")
print("  gdown 'YOUR_FILE_ID' -O /workspace/external_images.zip")
print("  unzip /workspace/external_images.zip -d /workspace/Data/")
print("  # Images will be at /workspace/Data/external_images/images/{category_id}/")
