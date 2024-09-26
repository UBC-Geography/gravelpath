import toml
from pathlib import Path
from lighttable.image_looper import ImageLooper
from lighttable.particle_looper_filter_test import Particle_Filters
from lighttable.particle_extractor import Particle_Extractor

#set up logging
import logging
from datetime import datetime

config_files = Path("configs_to_run").rglob("*.toml")

# iterate over each config file
for cf in config_files:
    # load the config file
    c = toml.load(cf)

    # Output directory, create if it doesn't exist
    Path(c["output"]["path"]).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    #set up logging, will be one file per execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).parent / "logs" / f"{datetime.now()}.log"),],)   

    # # run image processing
    # Looper = ImageLooper(c)
    # Looper.run()

    # # # connect particles
    Analyser = Particle_Filters(c)
    Analyser.run()

    # extract the data
    # Extractor = Particle_Extractor(c)
    # Extractor.run()

    print(f"Finished processing {cf.name}")