import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import seaborn as sns

import os

from filter_data import get_runs_with_metrics
from wandb_dl import Data

def set_axis(ax, env_name):
    if "antmaze" in env_name or "AntMaze" in env_name:
        ax.set_xticks(np.linspace(0., 0.3, 4))
        ax.set_xticks(np.linspace(0., 0.3, 13), minor=True)
        ax.set_xlim(0., 0.3)

        ax.set_yticks(np.linspace(0., 1.0, 5))
        ax.set_ylim(0., 1.03)
    elif "binary" in env_name or "Sparse" in env_name:
        ax.set_xticks(np.linspace(0., 0.8, 5))
        ax.set_xticks(np.linspace(0., 0.8, 17), minor=True)
        ax.set_xlim(0., 0.8)

        ax.set_yticks(np.linspace(0., 1.0, 6))
        ax.set_ylim(0., 1.03)

    elif "Widow" in env_name or "COG" in env_name:
        ax.set_xticks(np.linspace(0., 0.3, 4))
        ax.set_xticks(np.linspace(0., 0.3, 19), minor=True)
        ax.set_xlim(0., 0.25)

        ax.set_yticks(np.linspace(0., 1.0, 6))
        ax.set_ylim(0., 1.03)


cog_conf_dict = {
    "Oracle": {"offline_relabel_type": "gt", "offline_ratio": 0.5, "use_rnd_offline": False, "use_rnd_online": False},
    "Ours": {"offline_relabel_type": "pred", "use_rnd_offline": True, "use_icvf": False, "offline_ratio": 0.5},
    "Online": {"bc_pretrain_rollin": 0.0, "use_rnd_online": False, "offline_ratio" : 0.0},
    "Online + RND": {"bc_pretrain_rollin": 0.0, "use_rnd_online": True, "offline_ratio" : 0.0},
    "MinR": {"offline_relabel_type": "min", "use_rnd_online": False},
    "Naive": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "use_icvf": False},
    "Naive + BC": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.01"},
    "BC + JSRL": {"bc_pretrain_rollin": 0.5, "bc_pretrain_steps": 100000, "offline_ratio": 0.0, "use_icvf": False},
}

cog_full_conf_dict = {
    "Oracle": {"offline_relabel_type": "gt", "offline_ratio": 0.5, "use_rnd_offline": False, "use_rnd_online": False},
    "Ours + ICVF": {"offline_relabel_type": "pred", "use_rnd_offline": True, "use_icvf": True, "offline_ratio": 0.5},
    "Ours": {"offline_relabel_type": "pred", "use_rnd_offline": True, "use_icvf": False, "offline_ratio": 0.5},
    "Naive + ICVF": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "use_icvf": True},
    "Naive": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "use_icvf": False},
}

adroit_conf_dict = {
    "Oracle": {"offline_relabel_type": "gt", "offline_ratio": 0.5, "use_rnd_offline": False, "use_rnd_online": False},
    "Ours": {"offline_relabel_type": "pred", "use_rnd_offline": True, "offline_ratio": 0.5, "reset_rm_every": -1},
    "Ours + Reset": {"offline_relabel_type": "pred", "use_rnd_offline": True, "offline_ratio": 0.5, "reset_rm_every": 1000},
    "Online": {"bc_pretrain_rollin": 0.0, "use_rnd_online": False, "offline_ratio" : 0.0},
    "Online + RND": {"bc_pretrain_rollin": 0.0, "use_rnd_online": True, "offline_ratio" : 0.0},
    "MinR": {"offline_relabel_type": "min", "use_rnd_online": False},
    "Naive": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "reset_rm_every": -1},
    "Naive + Reset": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "reset_rm_every": 1000},
    "Naive + BC": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.01", "reset_rm_every": -1},
    "BC + JSRL": {"bc_pretrain_rollin": 0.5, "bc_pretrain_steps": 100000, "offline_ratio": 0.0},
}


adroit_reset_conf_dict = {
    "Oracle": {"offline_relabel_type": "gt", "offline_ratio": 0.5, "use_rnd_offline": False, "use_rnd_online": False},
    "Ours": {"offline_relabel_type": "pred", "use_rnd_offline": True, "offline_ratio": 0.5, "reset_rm_every": -1},
    "Ours + Reset": {"offline_relabel_type": "pred", "use_rnd_offline": True, "offline_ratio": 0.5, "reset_rm_every": 1000},
    "Naive": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "reset_rm_every": -1},
    "Naive + Reset": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0", "reset_rm_every": 1000},
}

antmaze_conf_dict = {
    "Oracle": {"offline_relabel_type": "gt", "offline_ratio": 0.5, "use_rnd_offline": False, "use_rnd_online": False},
    "Ours": {"offline_relabel_type": "pred", "use_rnd_offline": True, "offline_ratio": 0.5},
    "Online": {"bc_pretrain_rollin": 0.0, "use_rnd_online": False, "offline_ratio" : 0.0},
    "Online + RND": {"bc_pretrain_rollin": 0.0, "use_rnd_online": True, "offline_ratio" : 0.0},
    "MinR": {"offline_relabel_type": "min", "use_rnd_online": False},
    "Naive": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.0"},
    "Naive + BC": {"offline_relabel_type": "pred", "use_rnd_offline": False, "use_rnd_online": False, "config.bc_coeff": "0.01"},
    "BC + JSRL": {"bc_pretrain_rollin": 0.9, "bc_pretrain_steps": 5000, "offline_ratio": 0.0},
}


class PlotData:
    def __init__(self):
        self.data = {}

    def add(self, plot_name, line_name, steps, values):
        self.data[(plot_name, line_name)] = (steps, values)

    def get(self, plot_name, line_name):
        return self.data[(plot_name, line_name)]

    def plot_names(self):
        plot_names = []
        for plot_name, _ in self.data.keys():
            if plot_name not in plot_names:
                plot_names.append(plot_name)
        return plot_names

    def line_names(self):
        line_names = []
        for _, line_name in self.data.keys():
            if line_name not in line_names:
                line_names.append(line_name)
        return line_names

def make_plot(n_rows=2, n_cols=3, row_scale=2.4, col_scale=3.0, plot_data=None, colors=None, check_n=10, 
    ylabel="Normalized Return", save_path="default.pdf",
    legend_offset=1.25, legend_size=13, sharey=False, line_names=None, env_name=None, legend_ncol=4, labelsize=20,
):
    
    fig, axes = plt.subplots(n_rows, n_cols, 
        figsize=(col_scale * n_cols, row_scale * n_rows), dpi=200, sharey=sharey)

    if n_rows == 1 or n_cols == 1:
        if n_rows == n_cols == 1:
            axes = [axes]
    else:
        axes = [axes[j][i] for i in range(len(axes[0])) for j in range(len(axes))]
    
    subplot_names = plot_data.plot_names()
    if line_names is None:
        line_names = plot_data.line_names()
    for subplot_id, subplot_name in enumerate(subplot_names):

        if env_name is None:
            set_axis(axes[subplot_id], subplot_name)
        else:
            set_axis(axes[subplot_id], env_name)

        for line_id, line_name in enumerate(line_names):
            ax = axes[subplot_id]
            color = colors[line_id]

            steps, metric_values = plot_data.get(subplot_name, line_name)
            num_seeds = len(metric_values)
            assert num_seeds >= check_n
            steps = steps / 1000000.
            metrics_mean = np.mean(metric_values, axis=0)
            metrics_std = np.std(metric_values, axis=0) / (num_seeds ** 0.5)
            
            if line_name == "Oracle":
                linestyle = "--"
            else:
                linestyle = "-"

            ax.plot(steps, metrics_mean, label=line_name, color=color, alpha=0.9, linewidth=3.0, linestyle=linestyle)
            ax.fill_between(steps, metrics_mean - metrics_std, metrics_mean + metrics_std, color=color, alpha=0.15)

            ax.grid(color='lightgray', linewidth=1, alpha=0.5)
            ax.grid(color='lightgray', which="minor", linestyle="--", alpha=0.25)
            ax.spines['bottom'].set_color('lightgray')
            ax.spines['top'].set_color('lightgray') 
            ax.spines['right'].set_color('lightgray')
            ax.spines['left'].set_color('lightgray') 
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(f"{subplot_name} ({num_seeds})", fontsize=13)

            if len(subplot_names) == 1:
                ax.set_xlabel('Environment Steps ($\\times 10^6$)')
                ax.set_ylabel('Normalized Return')
        
    handles, labels = ax.get_legend_handles_labels()
    
    if len(subplot_names) > 1:
        fig.supxlabel('Environment Steps ($\\times 10^6$)', fontweight='bold', fontsize=labelsize)
        fig.supylabel(ylabel, fontweight='bold', fontsize=labelsize)

    fig.tight_layout()
    lgd = fig.legend(handles, labels, loc='upper center', ncol=legend_ncol, 
        bbox_to_anchor=(0.5, legend_offset), prop={'size': legend_size})

    fig.savefig(save_path, bbox_inches='tight')

def get_plot_data_antmaze_all(antmaze_raw_data, key="evaluation/return", partial=False, num_seeds=3):
    plot_data = PlotData()

    for env_name, display_name in zip([
        "antmaze-umaze-diverse-v2", "antmaze-umaze-v2", 
        "antmaze-medium-diverse-v2", "antmaze-medium-play-v2",
        "antmaze-large-diverse-v2", "antmaze-large-play-v2",
    ], ["umaze-diverse", "umaze", 
        "medium-diverse", "medium-play",
        "large-diverse", "large-play"]):
        if partial and "diverse" not in env_name: continue
        for method_name, conf in antmaze_conf_dict.items():
            conf["env_name"] = env_name
            conf["filter_data_mode"] = "all"
            # print(method_name, conf)
            steps, values, ids = get_runs_with_metrics(
                antmaze_raw_data, conf, 
                step_metric="env_step", 
                value_metric=key, 
                anc_steps=np.linspace(10000, 300000, 30), 
                patch_name="explore-antmaze-patch",
                extra_args="--max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1",
                min_length_v=20,
                num_seeds=num_seeds,
                is_pixel=False,
            )
            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")
            plot_data.add(
                plot_name=display_name,
                line_name=method_name,
                steps=steps,
                values=values,
            )
    return plot_data

def get_plot_data_adroit_all(adroit_raw_data, num_seeds=3):
    plot_data = PlotData()

    for env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
        for method_name, conf in adroit_conf_dict.items():
            
            if method_name == "Ours + Reset" and env_name != "relocate-binary-v0": continue
            if method_name == "Naive + Reset" and env_name != "relocate-binary-v0": continue
            if method_name == "Ours" and env_name == "relocate-binary-v0": continue
            if method_name == "Naive" and env_name == "relocate-binary-v0": continue
            
            conf["env_name"] = env_name
            steps, values, ids = get_runs_with_metrics(
                adroit_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 750000, 75), 
                patch_name="explore-adroit-patch-10-17",
                extra_args="--max_steps=1000000 --config.backup_entropy=False",
                min_length_v=75,
                num_seeds=num_seeds,
                is_pixel=False,
            )

            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")
            
            if method_name == "Ours + Reset": method_name = "Ours"
            if method_name == "Naive + Reset": method_name = "Naive"
            
            plot_data.add(
                plot_name=env_name,
                line_name=method_name,
                steps=steps,
                values=values,
            )
    return plot_data


def get_plot_data_adroit_reset(adroit_raw_data, num_seeds=3):
    plot_data = PlotData()

    env_name = "relocate-binary-v0"
    for method_name, conf in adroit_reset_conf_dict.items():
        
        conf["env_name"] = env_name
        steps, values, _ = get_runs_with_metrics(
            adroit_raw_data, conf, 
            step_metric="env_step", 
            value_metric="evaluation/return", 
            anc_steps=np.linspace(10000, 750000, 75), 
            patch_name="explore-adroit-patch-10-17",
            extra_args="--max_steps=1000000 --config.backup_entropy=False",
            min_length_v=75,
            num_seeds=num_seeds,
            is_pixel=False,
        )

        num = len(values)
        print(f"  {env_name} - {method_name}: {num} seeds")
        
        plot_data.add(
            plot_name=env_name,
            line_name=method_name,
            steps=steps,
            values=values,
        )
    return plot_data


def get_plot_data_cog_all(cog_raw_data, subsampling_ratio=0.1, num_seeds=3):
    
    plot_data = PlotData()
    for env_name, display_env_name in zip(
        ["Widow250PickTray-v0", "Widow250DoubleDrawerOpenGraspNeutral-v0", "Widow250DoubleDrawerCloseOpenGraspNeutral-v0"],
        ["Pick and Place", "Grasp from Closed Drawer", "Grasp from Blocked Drawer 1"]):
        for method_name, conf in cog_full_conf_dict.items():
            conf["env_name"] = env_name
            conf["dataset_subsample_ratio"] = subsampling_ratio
            steps, values, _ = get_runs_with_metrics(
                cog_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 250000, 25), 
                min_length_v=25,
                patch_name="explore-cog-patch",
                extra_args="--max_steps=500000",
                is_pixel=True,
                num_seeds=num_seeds,
            )
            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")

            plot_data.add(
                plot_name=display_env_name,
                line_name=method_name,
                steps=steps,
                values=values,
            )
    return plot_data


def get_plot_data_cog_all_patch(cog_raw_data, subsampling_ratio=0.1, num_seeds=3):
    
    plot_data = PlotData()
    for env_name, display_env_name in zip(
        ["Widow250PickTray-v0", "Widow250DoubleDrawerOpenGraspNeutral-v0", "Widow250DoubleDrawerCloseOpenGraspNeutral-v0"],
        ["Pick and Place", "Grasp from Closed Drawer", "Grasp from Blocked Drawer 1"]):
        for method_name, conf in cog_conf_dict.items():
            conf["env_name"] = env_name
            conf["dataset_subsample_ratio"] = subsampling_ratio
            steps, values, ids = get_runs_with_metrics(
                cog_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 250000, 25),
                min_length_v=25, 
                patch_name="explore-cog-patch",
                extra_args="--max_steps=500000",
                is_pixel=True,
                num_seeds=num_seeds,
            )
            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")

            plot_data.add(
                plot_name=display_env_name,
                line_name=method_name,
                steps=steps,
                values=values,
            )
    return plot_data

def get_agg_plot_data(antmaze_raw_data, adroit_raw_data, cog_raw_data, num_seeds=3, num_cog_seeds=5):

    plot_data = PlotData()

    for method_name, conf in antmaze_conf_dict.items():
        values_list = []
        for env_name, _ in zip([
            "antmaze-umaze-diverse-v2", "antmaze-umaze-v2", 
            "antmaze-medium-diverse-v2", "antmaze-medium-play-v2",
            "antmaze-large-diverse-v2", "antmaze-large-play-v2",
        ], ["umaze-diverse", "umaze", 
            "medium-diverse", "medium-play",
            "large-diverse", "large-play"]):
        
            conf["env_name"] = env_name
            conf["filter_data_mode"] = "all"
            steps, values, _ = get_runs_with_metrics(
                antmaze_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 300000, 30), 
                patch_name="explore-antmaze-patch",
                extra_args="--max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1",
                min_length_v=20,
                num_seeds=num_seeds,
                is_pixel=False,
            )
            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")
            values_list.append(values)

        plot_data.add(
            plot_name="AntMaze (6 Tasks)",
            line_name=method_name,
            steps=steps,
            values=np.stack(values_list, axis=0).mean(axis=0),
        )

    values_list = defaultdict(list)
    for method_name, conf in adroit_conf_dict.items():
        
        for env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
            
            if method_name == "Ours + Reset" and env_name != "relocate-binary-v0": continue
            if method_name == "Naive + Reset" and env_name != "relocate-binary-v0": continue
            if method_name == "Ours" and env_name == "relocate-binary-v0": continue
            if method_name == "Naive" and env_name == "relocate-binary-v0": continue
            
            conf["env_name"] = env_name
            steps, values, ids = get_runs_with_metrics(
                adroit_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 750000, 75), 
                patch_name="explore-adroit-patch-10-17",
                extra_args="--max_steps=1000000 --config.backup_entropy=False",
                min_length_v=75,
                num_seeds=num_seeds,
                is_pixel=False,
            )

            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")
            
            method_name_save = method_name
            if method_name == "Ours + Reset": method_name_save = "Ours" 
            if method_name == "Naive + Reset": method_name_save = "Naive"
            values_list[method_name_save].append(values)

    for method_name, conf in adroit_conf_dict.items():
        if "Reset" in method_name: continue
        plot_data.add(
            plot_name="Sparse Adroit (3 Tasks)",
            line_name=method_name,
            steps=steps,
            values=np.stack(values_list[method_name], axis=0).mean(axis=0),
        )

    for method_name, conf in cog_conf_dict.items():
        values_list = []
        for env_name, _ in zip(
            ["Widow250PickTray-v0", "Widow250DoubleDrawerOpenGraspNeutral-v0", "Widow250DoubleDrawerCloseOpenGraspNeutral-v0"],
            ["Pick and Place", "Grasp from Closed Drawer", "Grasp from Blocked Drawer 1"]):
            
            conf["env_name"] = env_name
            conf["dataset_subsample_ratio"] = 0.1
            steps, values, ids = get_runs_with_metrics(
                cog_raw_data, conf, 
                step_metric="env_step", 
                value_metric="evaluation/return", 
                anc_steps=np.linspace(10000, 250000, 25),
                min_length_v=25, 
                patch_name="explore-cog-patch",
                extra_args="--max_steps=500000",
                num_seeds=num_cog_seeds,
                is_pixel=True,
            )
            num = len(values)
            print(f"  {env_name} - {method_name}: {num} seeds")
            values_list.append(values)
        
        plot_data.add(
            plot_name="COG (3 Tasks)",
            line_name=method_name,
            steps=steps,
            values=np.stack(values_list, axis=0).mean(axis=0),
        )
    return plot_data

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    import argparse 
    parser = argparse.ArgumentParser(
        prog='Plot results',
        description='Generate paper plots'
    )

    parser.add_argument('--domain', type=str, help="one of [antmaze, adroit, cog, all]") 
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--num_cog_seeds', type=int, default=3)
    args = parser.parse_args()

    if args.domain == "cog" or args.domain == "all":
        with open('data/cog.pkl', 'rb') as handle:
            cog_raw_data = pkl.load(handle)

        plot_data = get_plot_data_cog_all(cog_raw_data, subsampling_ratio=0.1, num_seeds=args.num_cog_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=1, n_cols=3, row_scale=2.4, col_scale=3.0, plot_data=plot_data, colors=colors, 
            save_path="figures/cog-full-nss.pdf", check_n=args.num_cog_seeds,
            legend_size=15,
            legend_offset=1.25,
            legend_ncol=5,
            labelsize=16)
        
        plot_data = get_plot_data_cog_all_patch(cog_raw_data, subsampling_ratio=0.1, num_seeds=args.num_cog_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=1, n_cols=3, row_scale=2.4, col_scale=3.0, plot_data=plot_data, colors=colors, 
            save_path="figures/cog-full.pdf", 
            check_n=args.num_cog_seeds,
            legend_size=13,
            legend_offset=1.35,
            legend_ncol=4,
            labelsize=13,)

    if args.domain == "antmaze" or args.domain == "all":
        # AntMaze Full
        with open('data/antmaze.pkl', 'rb') as handle:
            antmaze_raw_data = pkl.load(handle)

        plot_data = get_plot_data_antmaze_all(antmaze_raw_data, num_seeds=args.num_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=2, n_cols=3, row_scale=2.4, col_scale=3.0, plot_data=plot_data, 
            colors=colors, 
            save_path="figures/antmaze-full.pdf", check_n=args.num_seeds,
            legend_size=15,
            legend_offset=1.15,
            sharey=True,
            env_name="antmaze"
        )

        plot_data = get_plot_data_antmaze_all(antmaze_raw_data, "coverage", num_seeds=args.num_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=2, n_cols=3, row_scale=2.4, col_scale=3.0, plot_data=plot_data, 
            colors=colors, 
            save_path="figures/antmaze-coverage-full.pdf", check_n=args.num_seeds,
            legend_size=15,
            legend_offset=1.15,
            sharey=True,
            ylabel="Coverage",
            env_name="antmaze",
            labelsize=16,
        )

        plot_data = get_plot_data_antmaze_all(antmaze_raw_data, "coverage", partial=True, num_seeds=args.num_seeds)
        make_plot(n_rows=1, n_cols=3, row_scale=2.3, col_scale=2.7, plot_data=plot_data, 
            colors=sns.color_palette("muted", 3), 
            save_path="figures/antmaze-coverage-main.pdf", check_n=args.num_seeds,
            legend_size=13,
            legend_offset=1.15,
            sharey=True,
            ylabel="Coverage",
            line_names=["Ours", "Naive", "Online + RND"],
            env_name="antmaze",
            labelsize=16,
        )

    if args.domain == "adroit" or args.domain == "all":
        # Adroit Full
        with open('data/adroit.pkl', 'rb') as handle:
            adroit_raw_data = pkl.load(handle)

        plot_data = get_plot_data_adroit_all(adroit_raw_data, num_seeds=args.num_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=1, n_cols=3, row_scale=3.0, col_scale=3.0, plot_data=plot_data, 
            colors=colors, save_path="figures/adroit-full.pdf", check_n=args.num_seeds,
            legend_size=15,
            legend_offset=1.25,
            sharey=True,
        )

        plot_data = get_plot_data_adroit_reset(adroit_raw_data, num_seeds=args.num_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=1, n_cols=1, row_scale=3.0, col_scale=2.5, plot_data=plot_data, 
            colors=colors, save_path="figures/adroit-reset.pdf", check_n=args.num_seeds,
            legend_size=10,
            legend_offset=1.25,
            legend_ncol=2,
            sharey=True,
        )

    if args.domain == "all":
        plot_data = get_agg_plot_data(antmaze_raw_data, adroit_raw_data, cog_raw_data, num_seeds=args.num_seeds, num_cog_seeds=args.num_cog_seeds)

        colors = ["black"] + sns.color_palette("muted", len(plot_data.line_names()))
        make_plot(n_rows=1, n_cols=3, row_scale=3.0, col_scale=2.5, plot_data=plot_data, 
            colors=colors, save_path="figures/main-agg.pdf", check_n=args.num_seeds,
            legend_size=15,
            legend_offset=1.25,
            legend_ncol=4,
            labelsize=16,
            sharey=True,
        )
