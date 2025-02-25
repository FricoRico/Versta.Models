from versta.export import __main__ as export
from versta.bundle import __main__ as bundle
from pathlib import Path
from json import load

with open("models.json", "r") as f:
    models = load(f)

for entry in models:
    export.main(
        model=entry["model"],
        output_dir=Path("output"),
        temp_dir=Path("tmp"),
    )