#! /usr/bin/env python
# coding: utf-8

# # Base class for a grid-based Landlab model
#
# *(Greg Tucker, University of Colorado Boulder)*
#

import sys
import time

import numpy as np
from landlab import ModelGrid, create_grid, load_params

from landlab.io.native_landlab import load_grid, save_grid



def merge_user_and_default_params(user_params, default_params):
    """Merge default parameters into the user-parameter dictionary, adding
    defaults where user values are absent.

    Examples
    --------
    >>> u = {"a": 1, "d": {"da": 4}, "e": 5, "grid": {"RasterModelGrid": []}}
    >>> d = {"a": 2, "b": 3, "d": {"db": 6}, "grid": {"HexModelGrid": []}}
    >>> merge_user_and_default_params(u, d)
    >>> u["a"]
    1
    >>> u["b"]
    3
    >>> u["d"]
    {'da': 4, 'db': 6}
    >>> u["grid"]
    {'RasterModelGrid': []}
    """
    for k in default_params.keys():
        if k in default_params:
            if k not in user_params.keys():
                user_params[k] = default_params[k]
            elif isinstance(user_params[k], dict) and k != "grid":
                merge_user_and_default_params(user_params[k], default_params[k])


def get_or_create_node_field(grid, name, dtype="float64"):
    """Get handle to a grid field if it exists, otherwise create it."""
    try:
        return grid.at_node[name]
    except KeyError:
        return grid.add_zeros(name, at="node", dtype=dtype, clobber=True)


class LandlabModel:
    """Base class for a generic Landlab grid-based model."""

    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    {"shape": (5, 5)}, {"spacing": 1.0},
                ],
            },
        },
        "clock": {"start": 0.0, "stop": 2.0, "step": 1.0},
        "output": {
            "interval": 10,
            "filepath": "model_output",
            "clobber": True,
            "fields": None,
        },
    }

    def __init__(
        self, params
    ):
        """Initialize the model."""

        self.setup_grid(params["grid"])
        self.setup_fields()
        self.setup_for_output(params["output"])
        self.instantiate_components(params)
        self.setup_run_control(params)

    def setup_grid(self, grid_params):
        """Load or create the grid.

        Examples
        --------
        >>> p = {"grid": {"source": "create"}}
        >>> p["grid"]["create_grid"] = {"RasterModelGrid": {"shape": (4, 5), "xy_spacing": 2.0}}
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab.io.native_landlab import save_grid
        >>> save_grid(sim.grid, "test.grid", clobber=True)
        >>> p = {"grid": {"source": "file", "grid_file_name": "test.grid"}}
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab import RasterModelGrid
        >>> p = {"grid": {"source": "grid_object"}}
        >>> p["grid"]["grid_object"] = RasterModelGrid((3, 3))
        >>> sim = LandlabModel(params=p)
        >>> sim.grid.shape
        (3, 3)
        >>> from numpy.testing import assert_raises
        >>> p["grid"]["grid_object"] = "spam"
        >>> assert_raises(ValueError, LandlabModel, p)
        grid_object must be a Landlab grid.
        """
        if grid_params["source"] == "create":
            self.grid = create_grid(grid_params, section="create_grid")
        elif grid_params["source"] == "file":
            self.grid = load_grid(grid_params["grid_file_name"])
        elif grid_params["source"] == "grid_object":
            if isinstance(grid_params["grid_object"], ModelGrid):
                self.grid = grid_params["grid_object"]
            else:
                print("grid_object must be a Landlab grid.")
                raise ValueError

    def setup_for_output(self, params):
        """Setup variables for control of plotting and saving."""
        self.plot_interval = params["plot_interval"]
        self.next_plot = self.plot_interval
        self.save_interval = params["save_interval"]
        self.next_save = self.save_interval
        self.ndigits = params["ndigits"]
        self.frame_num = 0  # current output image frame number
        self.save_num = 0  # current save file frame number
        self.save_name = params["save_name"]
        self.display_params = params

    def setup_run_control(self, params):
        """Initialize variables related to control of run timing."""
        self.run_duration = params["run_duration"]
        self.dt = params["dt"]
        self.current_time = params["start_time"]

    def update(self, dt):
        """Advance the model by one time step of duration dt."""
        self.current_time += dt

    def update_until(self, update_to_time, dt):
        """Iterate up to given time, using time-step duration dt."""
        remaining_time = update_to_time - self.current_time
        while remaining_time > 0.0:
            dt = min(dt, remaining_time)
            self.update(dt)
            remaining_time -= dt

    def run(self, run_duration=None, dt=None):
        """Run the model for given duration, or self.run_duration if none given.

        Includes file output of images and model state at user-specified
        intervals.
        """
        if run_duration is None:
            run_duration = self.run_duration
        if dt is None:
            dt = self.dt

        stop_time = run_duration + self.current_time
        while self.current_time < stop_time:
            next_pause = min(self.next_plot, self.next_save)
            next_pause = min(next_pause, self.next_report)
            self.update_until(next_pause, dt)
            if self.current_time >= self.next_report:
                self.report()
            if self.current_time >= self.next_plot:
                self.plot()
                self.next_plot += self.plot_interval
            if self.current_time >= self.next_save:
                self.save_num += 1
                self.save_state(self.save_num)


if __name__ == "__main__":
    """Launch a run.

    Optional command-line argument is the name of a yaml-format text file with
    parameters. File should include sections for "grid_setup", "process",
    "run_control", and "output". Each of these should have the format shown in
    the defaults defined above in the class header.
    """
    if len(sys.argv) > 1:
        params = load_params(sys.argv[1])
        sim = IslandSimulator(
            params["grid_setup"],
            params["process"],
            params["run_control"],
            params["output"],
        )
    else:
        sim = IslandSimulator()  # use default params
    sim.run()
