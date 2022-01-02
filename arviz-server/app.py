import enum
import os
import time
from typing import Sequence

import arviz
import clickhouse_driver
import numpy
import streamlit as st
from matplotlib import pyplot

import mcbackend
from mcbackend.backends.clickhouse import ClickHouseChain, ClickHouseRun

ACTIONS = {
    "plot_trace": arviz.plot_trace,
    "plot_pair": arviz.plot_pair,
}


class App:
    def __init__(self) -> None:
        st.title("Welcome to ArviZ Server")

        client = clickhouse_driver.Client(os.getenv("ARVIZ_DB_HOST", "localhost"))
        self.backend = mcbackend.ClickHouseBackend(client)
        self.df_runs = self.backend.get_runs()
        st.write(f"✔ ClickHouse connection established. {len(self.df_runs)} runs found.")
        self.selected_actions = ["plot_trace"]
        super().__init__()

    def run(self):
        run, chains = self.select_run()

        self.selected_actions = st.multiselect(
            label="Pick ArviZ visualizations", options=ACTIONS.keys(), default=self.selected_actions
        )

        st.button("Refresh", on_click=self.refresh(run, chains))
        return

    def refresh(self, run, chains):
        idata = self.refresh_idata(run, chains)
        self.make_plots(idata, self.selected_actions)
        return

    def select_run(self):
        if len(self.df_runs) == 0:
            st.write("There are no MCMC runs in the database yet.")
            st.stop()
        rid = st.text_input(f"What's your MCMC run ID?", placeholder=self.df_runs.index[-1])
        if not rid:
            st.stop()
        run = self.backend.get_run(rid)
        chains = run.get_chains()
        # st.write(f"✔ Found {len(chain_ids)} chains for run {rid}.")
        return run, chains

    def refresh_idata(self, run: ClickHouseRun, chains: Sequence[ClickHouseChain]):
        t_start = time.time()
        nchains = len(chains)
        var_names = [var.name for var in run.meta.variables if not var.is_deterministic]
        var_shapes = [var.shape for var in run.meta.variables if not var.is_deterministic]

        # Prepare dims and coords from run metadata
        dims = {}
        coords = {
            "chain": [c.cmeta.chain_number for c in chains],
            "draw": None,  # this is unknown until chains were loaded
        }
        for var in run.meta.variables:
            vardims = var.dims or [f"{var.name}_dim_{d}" for d, s in enumerate(var.shape)]
            for dname, s in zip(vardims, var.shape):
                coords[dname] = list(range(s))
            dims[var.name] = ["chain", "draw"] + vardims

        # Now load MCMC draws for each chain
        samples = {vn: [] for vn in var_names}
        for vn in var_names:
            for chain in chains:
                vals = chain.get_variable(vn)
                samples[vn].append(vals)
        # Now cut them to the same length
        max_length = min(len(samples[vn][c]) for vn in var_names for c in range(nchains))
        samples = {
            vn: numpy.array([chaindraws[c][:max_length] for c in range(nchains)])
            for vn, chaindraws in samples.items()
        }
        coords["draw"] = list(range(max_length))

        try:
            idata = arviz.from_dict(
                posterior=samples,
                coords=coords,
                dims=dims,
            )
        except:
            st.write(max_length)
            st.write(dims)
            st.write(coords)
            for k, v in samples.items():
                st.write(f"{k}: {numpy.shape(v)}")
        st.write(
            f"✔ Loaded {nchains} chains x {max_length} draws in {time.time() - t_start:.3f} seconds."
        )
        return idata

    def make_plots(self, idata, actions):
        for name in actions:
            st.write(f"Running {name}...")
            fun = ACTIONS[name]
            fun(idata)
            fig = pyplot.gcf()
            fig.tight_layout()
            st.pyplot(fig)
            pyplot.close()
        return


if __name__ == "__main__":
    app = App()
    app.run()
