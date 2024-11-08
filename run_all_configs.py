import toml
from pathlib import Path
from lighttable.image_looper import ImageLooper
from lighttable.particle_looper_filter_test import Particle_Filters
from lighttable.particle_extractor import Particle_Extractor

# set up logging
import logging
from datetime import datetime
import os

# set up multiprocessing
import numpy as np
import multiprocessing as mp

def chunk(images, numImagesPerProc):
    """
    Break image paths into separate chunks that can be served to each process
    """
    for i in range(0, len(images), numImagesPerProc):
        yield images[i : i + numImagesPerProc]


if __name__ == "__main__":
    # Ensure multiprocessing works the same on all systems
    mp.set_start_method("spawn")
    # Get the max number of cpu cores / threads to run the work on
    procs = mp.cpu_count()

    config_files = Path("configs_to_run").rglob("*.toml")

    # iterate over each config file
    for cf in config_files:
        # load the config file
        c = toml.load(cf)

        # Output directory, create if it doesn't exist
        Path(c["output"]["path"]).mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)

        # mkdir logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # set up logging, will be one file per execution
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(__file__).parent / "logs" / f"{datetime.now()}.log"
                ),
            ],
        )

        # chunk the images into separate payloads, which can be passed to each processor
        images = list(Path(c["images"]["path"]).rglob("*.tif"))
        images = sorted(images)
        numImagesPerProc = len(images) / float(procs)
        numImagesPerProc = int(np.ceil(numImagesPerProc))
        chunkedPaths = list(chunk(images, numImagesPerProc))
        payloads = []

        for i, images in enumerate(chunkedPaths):
            data = {
                "id": i,
                "image_paths": images,
            }
            payloads.append(data)

        # # run image processing
        Looper = ImageLooper(c)
        Looper.run(payloads)

        # # # connect particles
        Analyser = Particle_Filters(c)
        Analyser.run()

        # extract the data
        Extractor = Particle_Extractor(c)
        Extractor.run()

        print(f"Finished processing {cf.name}")
