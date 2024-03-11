import toml
from pathlib import Path
from lighttable.image_looper import ImageLooper

config_files = Path("configs_to_run").rglob("*.toml")

# iterate over each config file
for cf in config_files:
    # load the config file
    c = toml.load(cf)

    # Output directory, create if it doesn't exist
    Path(c["output"]["path"]).mkdir(parents=True, exist_ok=True)

    # run image processing
    Looper = ImageLooper(c)
    Looper.run()
    print(f"Finished processing {cf.name}")
