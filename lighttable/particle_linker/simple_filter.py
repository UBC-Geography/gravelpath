# module that contains a routine to link particles between two frames
import pandas as pd
from matplotlib import pyplot as plt


class ParticleLinker:
    def __init__(self, config):
        self.config = config

        self.df = pd.DataFrame()

    def run(self, df_prev, df_now):

        df_out = df_now.copy()

        # display the particles
        for index, row_now in df_now.iterrows():

            df_prev_filtered = df_prev.copy()

            # filter out all particles that are less than half or double the area of the current particle
            df_prev_filtered = df_prev_filtered[
                (df_prev_filtered["area"] > row_now["area"] / 2)
                & (df_prev_filtered["area"] < row_now["area"] * 2)
            ]

            # filter out all previous particles that have a lower y coordinate than the current particle
            df_prev_filtered = df_prev_filtered[df_prev_filtered["y"] > row_now["y"]]

            # filter out all previous particles that are translated to far left or right
            # TODO: make this value configurable
            df_prev_filtered = df_prev_filtered[
                (df_prev_filtered["x"] > row_now["x"] - 50)
                & (df_prev_filtered["x"] < row_now["x"] + 50)
            ]

            # cost based on area difference
            df_prev_filtered["cost_area_diff"] = (
                abs(df_prev_filtered["area"] - row_now["area"]) / row_now["area"]
            )

            # cost based on x difference
            df_prev_filtered["cost_x_diff"] = (
                abs(df_prev_filtered["x"] - row_now["x"]) / row_now["x"]
            )

            # cost based on y difference (lower)
            df_prev_filtered["cost_y_diff"] = (
                abs(df_prev_filtered["y"] - row_now["y"]) / row_now["y"]
            )

            # sum the costs
            df_prev_filtered["cost"] = (
                df_prev_filtered["cost_area_diff"] * 2
                + df_prev_filtered["cost_x_diff"] * 10
                + df_prev_filtered["cost_y_diff"]
            )

            df_prev_filtered.sort_values("cost", ascending=True, inplace=True)

            if df_prev_filtered.empty:
                df_out.loc[index, "prev_index"] = -1
                continue

            # add the best match to the output
            # TODO: maybe run a second pass to ensure that particles are not double counted
            df_out.loc[index, "prev_index"] = df_prev_filtered.index[0]

        return df_out
