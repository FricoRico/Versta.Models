from versta.export import __main__ as export
from versta.bundle import __main__ as bundle
from pathlib import Path
from json import load

def main():
    with open("models.json", "r") as f:
        models = load(f)

    for pair in models:
        exported_models: list[Path] = []

        for entry in pair:
            exported_models.append(export.main(
                model=entry["model"],
                output_dir=Path("output"),
                clear_cache=True,
            ))

        bundle.main(
            input_dirs=exported_models,
            output_dir=Path("output"),
            bidirectional=len(exported_models) > 1,
        )

if __name__ == "__main__":
    main()