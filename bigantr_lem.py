#! /usr/bin/env python
# coding: utf-8

# # BIGANTR LEM: Landscape Evolution Model using theory for Bedrock-Incising,
#   Gravel-Abrading, Near-Threshold Rivers
#
# *(Greg Tucker, University of Colorado Boulder)*
#

import sys

import numpy as np
from landlab.core import load_params
from landlab.components import GravelBedrockEroder, PriorityFloodFlowRouter

from model_base import LandlabModel


class BigantrLEM(LandlabModel):
    """Landscape Evolution Model using fluvial BIGANTR theory."""

    DEFAULT_PARAMS = {
        "grid": {
            "source": "create",
            "create_grid": {
                "RasterModelGrid": [
                    (31, 31),
                    {"xy_spacing": 1000.0},
                ],
            },
        },
        "clock": {"start": 0.0, "stop": 10000.0, "step": 10.0},
        "output": {
            "plot_times": 2000.0,  # float or list
            "save_times": 10000.0,  # float or list
            "report_times": 1000.0,  # float or list
            "save_path": "bigantr_run",
            "clobber": True,
            "fields": None,
            "plot_to_file": True,
        },
        "initial_conditions": {
            "initial_sed_thickness": 1.0,
            "random_topo_amp": 10.0,
        },
        "baselevel": {
            "uplift_rate": 0.0001,
        },
        "flow_routing": {
            "flow_metric": "D8",
            "update_flow_depressions": True,
            "depression_handler": "fill",
            "epsilon": True,
        },
        "fluvial": {
            "intermittency_factor": 0.01,
            "transport_coefficient": 0.041,
            "sediment_porosity": 1.0 / 3.0,
            "depth_decay_scale": 1.0,
            "plucking_coefficient": 1.0e-4,
            "number_of_sediment_classes": 1,
            "init_thickness_per_class": [1.0],
            "abrasion_coefficients": [1.0e-4],
            "coarse_fractions_from_plucking": [0.5],
            "rock_abrasion_index": 0,
        },
    }

    def __init__(self, params={}):
        """Initialize the model."""
        super().__init__(params)

        # Set up grid fields
        ic_params = params["initial_conditions"]
        if not ("topographic__elevation" in self.grid.at_node.keys()):
            self.grid.add_zeros("topographic__elevation", at="node")
            self.grid.at_node["topographic__elevation"][
                self.grid.core_nodes
            ] += ic_params["random_topo_amp"] * np.random.rand(
                self.grid.number_of_core_nodes
            )
        if not ("soil__depth" in self.grid.at_node.keys()):
            self.grid.add_zeros("soil__depth", at="node")
            self.grid.at_node["soil__depth"][:] = ic_params["initial_sed_thickness"]
        self.topo = self.grid.at_node["topographic__elevation"]
        self.sed = self.grid.at_node["soil__depth"]

        # Store parameters
        self.uplift_rate = params["baselevel"]["uplift_rate"]

        # Instantiate and initialize components: flow router
        flow_params = params["flow_routing"]
        self.router = PriorityFloodFlowRouter(
            self.grid,
            surface="topographic__elevation",
            flow_metric=flow_params["flow_metric"],
            update_flow_depressions=flow_params["update_flow_depressions"],
            depression_handler=flow_params["depression_handler"],
            epsilon=flow_params["epsilon"],
            accumulate_flow=True,
        )

        # Instantiate and initialize components: fluvial transport, erosion, deposition
        gbe_params = params["fluvial"]
        self.eroder = GravelBedrockEroder(
            self.grid,
            intermittency_factor=gbe_params["intermittency_factor"],
            transport_coefficient=gbe_params["transport_coefficient"],
            sediment_porosity=gbe_params["sediment_porosity"],
            depth_decay_scale=gbe_params["depth_decay_scale"],
            plucking_coefficient=gbe_params["plucking_coefficient"],
            number_of_sediment_classes=gbe_params["number_of_sediment_classes"],
            init_thickness_per_class=gbe_params["init_thickness_per_class"],
            abrasion_coefficients=gbe_params["abrasion_coefficients"],
            coarse_fractions_from_plucking=gbe_params["coarse_fractions_from_plucking"],
            rock_abrasion_index=gbe_params["rock_abrasion_index"],
        )

    def update(self, dt):
        """Advance the model by one time step of duration dt."""
        self.topo[self.grid.core_nodes] += self.uplift_rate * dt
        self.router.run_one_step()
        self.eroder.run_one_step(dt)
        self.current_time += dt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        params = load_params(sys.argv[1])
    else:
        params = {}
    bigantr = BigantrLEM(params)
    bigantr.run()
