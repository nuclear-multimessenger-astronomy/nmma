from os.path import abspath

from .parser import create_nmma_analysis_parser
from parallel_bilby.utils import get_cli_args

from .._version import __version__
__prog__ = "nmma_analysis"


def setup_submit(data_dump_file, inputs, args):

    # Create analysis nodes
    analysis_nodes = []
    for idx in range(args.n_parallel):
        node = AnalysisNode(data_dump_file, inputs, idx, args)
        node.write()
        analysis_nodes.append(node)

    if len(analysis_nodes) > 1:
        final_analysis_node = MergeNodes(analysis_nodes, inputs, args)
        final_analysis_node.write()
    else:
        final_analysis_node = analysis_nodes[0]

    bash_script = "{}/bash_{}.sh".format(inputs.submit_directory, inputs.label)
    with open(bash_script, "w+") as ff:
        dependent_job_ids = []
        for ii, node in enumerate(analysis_nodes):
            print("jid{}=$(sbatch {})".format(ii, node.filename), file=ff)
            dependent_job_ids.append("${{jid{}##* }}".format(ii))
        if len(analysis_nodes) > 1:
            print(
                "sbatch --dependency=afterok:{} {}".format(
                    ":".join(dependent_job_ids), final_analysis_node.filename
                ),
                file=ff,
            )
        print('squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"', file=ff)

    return bash_script


class BaseNode(object):
    def get_lines(self):
        lines = ["#!/bin/bash"]
        lines.append("#SBATCH --job-name={}".format(self.job_name))
        if self.nodes > 1:
            lines.append("#SBATCH --nodes={}".format(self.nodes))
        if self.ntasks_per_node > 1:
            lines.append("#SBATCH --ntasks-per-node={}".format(self.ntasks_per_node))
        lines.append("#SBATCH --time={}".format(self.time))
        if self.args.mem_per_cpu is not None:
            lines.append("#SBATCH --mem-per-cpu={}".format(self.mem_per_cpu))
        lines.append("#SBATCH --output={}/{}.log".format(self.logs, self.job_name))
        if self.args.slurm_extra_lines is not None:
            slurm_extra_lines = " ".join(
                ["--{}".format(lin) for lin in self.args.slurm_extra_lines.split()]
            )
            for line in slurm_extra_lines.split():
                lines.append("#SBATCH {}".format(line))
        lines.append("")
        if self.args.extra_lines:
            for line in self.args.extra_lines.split(";"):
                lines.append(line.strip())
        lines.append("")
        return lines

    def get_contents(self):
        lines = self.get_lines()
        return "\n".join(lines)

    def write(self):
        content = self.get_contents()
        with open(self.filename, "w+") as f:
            print(content, file=f)


class AnalysisNode(BaseNode):
    def __init__(self, data_dump_file, inputs, idx, args):
        self.data_dump_file = data_dump_file
        self.inputs = inputs
        self.args = args
        self.idx = idx
        self.filename = "{}/analysis_{}_{}.sh".format(
            self.inputs.submit_directory, self.inputs.label, self.idx
        )

        self.job_name = "{}_{}".format(self.idx, self.inputs.label)
        self.nodes = self.args.nodes
        self.ntasks_per_node = self.args.ntasks_per_node
        self.time = self.args.time
        self.mem_per_cpu = self.args.mem_per_cpu
        self.logs = self.inputs.data_analysis_log_directory

        # This are the defaults: used only to figure out which arguments to use
        analysis_parser = create_nmma_analysis_parser(__prog__, __version__)
        self.analysis_args, _ = analysis_parser.parse_known_args(args=get_cli_args())
        # hack -- in the above the parse_known_arg sets the position param (ini) as
        # the data dump
        self.analysis_args.data_dump = self.data_dump_file

    @property
    def executable(self):
        if self.args.sampler == "dynesty":
            return "nmma_analysis"
        else:
            raise ValueError(
                "Unable to determine sampler to use from {}".format(self.args.sampler)
            )

    @property
    def label(self):
        return "{}_{}".format(self.inputs.label, self.idx)

    @property
    def output_filename(self):
        return "{}/{}_{}_result.json".format(
            self.inputs.result_directory, self.inputs.label, self.idx
        )

    def get_contents(self):
        lines = self.get_lines()
        lines.append('export MKL_NUM_THREADS="1"')
        lines.append('export MKL_DYNAMIC="FALSE"')
        lines.append("export OMP_NUM_THREADS=1")
        lines.append("export MPI_PER_NODE={}".format(self.args.ntasks_per_node))
        lines.append("")

        run_string = self.get_run_string()
        lines.append("mpirun {} {}".format(self.executable, run_string))
        return "\n".join(lines)

    def get_run_string(self):
        run_list = ["{}".format(self.data_dump_file)]
        for key, val in vars(self.analysis_args).items():
            if key in ["data_dump", "label", "outdir", "sampling_seed"]:
                continue
            input_val = getattr(self.args, key)
            if val != input_val:
                if input_val is True:
                    # For flags only add the flag
                    run_list.append("--{}".format(key.replace("_", "-")))
                else:
                    run_list.append("--{} {}".format(key.replace("_", "-"), input_val))

        run_list.append("--label {}".format(self.label))
        run_list.append("--outdir {}".format(abspath(self.inputs.result_directory)))
        run_list.append(
            "--sampling-seed {}".format(self.analysis_args.sampling_seed + self.idx)
        )

        return " ".join(run_list)


class MergeNodes(BaseNode):
    def __init__(self, analysis_nodes, inputs, args):
        self.analysis_nodes = analysis_nodes

        self.inputs = inputs
        self.args = args
        self.job_name = "merge_{}".format(self.inputs.label)
        self.nodes = 1
        self.ntasks_per_node = 1
        self.time = "1:00:00"
        self.mem_per_cpu = "16GB"
        self.logs = self.inputs.data_analysis_log_directory

        self.filename = "{}/merge_{}.sh".format(
            self.inputs.submit_directory, self.inputs.label
        )

    @property
    def file_list(self):
        return " ".join([node.output_filename for node in self.analysis_nodes])

    @property
    def merged_result_label(self):
        return "{}_merged".format(self.inputs.label)

    def get_contents(self):
        lines = self.get_lines()
        lines.append(
            "bilby_result -r {} --merge --label {} --outdir {}".format(
                self.file_list, self.merged_result_label, self.inputs.result_directory
            )
        )
        return "\n".join(lines)
