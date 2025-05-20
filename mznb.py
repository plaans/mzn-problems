from __future__ import annotations
from argparse import ArgumentParser, ArgumentTypeError, Namespace, RawTextHelpFormatter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import logging
from pathlib import Path
import subprocess
import sys
from typing import Optional

# Command code for success
CODE_OK = 0


@dataclass(order=True, frozen=True)
class Instance:
    """Minizinc instance."""
    model: Path
    data: Optional[Path]

    def __str__(self) -> str:
        return f"{self.model}" if self.data is None else f"{self.model} {self.data}"

    def paths(self) -> list[Path]:
        """Return the path(s) making the instance."""
        return [self.model] if self.data is None else [self.model, self.data]

    @staticmethod
    def combine(models: list[Path], datas: list[Path]) -> list[Instance]:
        """Return a list of model-data pair or model-None if no dzn."""
        instances = []
        if not datas:
            datas = [None]
        for model in models:
            for data in datas:
                instance = Instance(model, data)
                instances.append(instance)
        return instances

    @staticmethod
    def collect(directory: Path, recursive: bool = False) -> list[Instance]:
        """Return all instances in the given directory."""
        instances = []
        if recursive:
            directories = filter(lambda d: d.is_dir(), directory.rglob("*"))
            instances.extend(Instance.collect(directory))
            for d in directories:
                instances.extend(Instance.collect(d))
        else:
            models = list(directory.glob("*.mzn"))
            datas = list(directory.glob("*.dzn"))
            models.sort()
            datas.sort()
            instances = Instance.combine(models, datas)
        return instances


class CmdStatus(Enum):
    ERROR = auto()
    TIMEOUT = auto()
    OK = auto()


class Driver:
    """Minizinc driver."""

    COMMENT_END_OF_SEARCH = "% =========="
    COMMENT_END_OF_SOLUTION = "% ----------"
    UNSAT = "=====UNSATISFIABLE====="

    def __init__(self, minzinc: str = "minizinc", include_dir: Optional[Path] = None, timeout: float = 1.0) -> None:
        """Create a new minizinc driver."""
        self.minizinc = minzinc
        self.include_dir = include_dir
        self.timeout = timeout
        self.base_command = self._base_command()

    def _base_command(self) -> list[str]:
        command = [self.minizinc]
        if self.include_dir is not None:
            command.extend(["-I", self.include_dir])
        return command

    def _log(self, command: list[str | Path]) -> None:
        """Log the given command."""
        message = " ".join(map(
            lambda a: f"'{a}'" if " " in a else a,
            map(str,command),
        ))
        logging.info(message)

    def _execute(self, command: list[str | Path]) -> tuple[CmdStatus, str]:
        """Execute the given command and return wether it succeeded and stderr."""
        self._log(command)
        try:
            process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
            _, stderr = process.communicate(timeout=self.timeout)
            stderr = "" if stderr is None else stderr
            status = CmdStatus.OK if process.returncode == CODE_OK else CmdStatus.ERROR
        except subprocess.TimeoutExpired as e:
            status = CmdStatus.TIMEOUT
            stderr = str(e)
        finally:
            process.terminate()
        return status, stderr

    def check_instance(self, instance: Instance) -> tuple[CmdStatus, str]:
        """Return wether the model is valid and stderr output."""
        command = self.base_command + [
            "--instance-check-only",
        ] + instance.paths()
        status, stderr = self._execute(command)
        return status, stderr

    def compile(self, instance: Instance, solver: str, output: Path) -> tuple[CmdStatus, str]:
        """Compile the given instance to flatzinc."""
        command = self.base_command + [
            "--solver", solver,
            "--compile",
            "--output-fzn-to-file", output,
            "--no-output-ozn"
        ] + instance.paths()
        status, stderr = self._execute(command)
        return status, stderr

    def solve(self, instance: Instance, solver: str, output: Path) -> tuple[CmdStatus, str]:
        """Solve the given instance and save solution in dzn format."""
        command = self.base_command + [
            "--solver", solver,
            "--output-mode", "dzn",
            "--soln-sep", Driver.COMMENT_END_OF_SOLUTION,
            "--search-complete-msg", Driver.COMMENT_END_OF_SEARCH,
            "-o", output,
        ] + instance.paths()
        status, stderr = self._execute(command)
        return status, stderr

    def check_solution(self, instance: Instance, solver: str, solution: Path, output: Path) -> tuple[CmdStatus, str]:
        """Check the given solution is valid by adding it to instance and solving."""

        solution_sat = solution.exists() and Driver.UNSAT not in solution.read_text()
        solution_opt = [solution] if solution_sat else []

        command = self.base_command + [
            "--solver", solver,
            "--output-mode", "dzn",
            "--soln-sep", Driver.COMMENT_END_OF_SOLUTION,
            "--search-complete-msg", Driver.COMMENT_END_OF_SEARCH,
            "-o", output,
        ] + instance.paths() + solution_opt
        status, stderr = self._execute(command)

        # If ok: check sat or unsat
        if status == CmdStatus.OK:
            check_sat = output.exists() and Driver.UNSAT not in output.read_text()
            if solution_sat and not check_sat:
                stderr = "Solver has found a solution but checker does not agree."
                status = CmdStatus.ERROR
            elif not solution_sat and check_sat:
                stderr = "Solver says UNSAT but checker has a solution."
                status = CmdStatus.ERROR

        return status, stderr


class Status(Enum):
    INSTANCE_ERROR = "INST-ERR"
    INSTANCE_TIMEOUT = "INST-TIM"
    COMPILATION_ERROR = "COMP-ERR"
    COMPILATION_TIMEOUT = "COMP-TIM"
    SOLVE_ERROR = "SOLV-ERR"
    SOLVE_TIMEOUT = "SOLV-TIM"
    CHECK_ERROR = "CHEC-ERR"
    CHECK_TIMEOUT = "CHEC-TIM"
    OK = "OK"

    def __str__(self) -> str:
        return self.value


def is_dzn(path: Path) -> bool:
    """Return wether the given path is a dzn."""
    return path.suffix == ".dzn"


def is_mzn(path: Path) -> bool:
    """Return wether the given path is a mzn."""
    return path.suffix == ".mzn"


def input_mzn_dzn_dir(s: str) -> Path:
    """Ensure the input is either a mzn/dzn file or a directory."""
    path = Path(s)
    if not path.exists():
        raise ArgumentTypeError(f"{s} not found")
    if path.is_file():
        if not is_mzn(path) and not is_dzn(path):
            raise ArgumentTypeError(f"{s} file does not have mzn/dzn extension")
    elif path.is_dir():
        if is_mzn(path) or is_dzn(path):
            raise ArgumentTypeError(f"{s} is a directory but has mzn/dzn extension")
    else:
        raise ArgumentTypeError(f"{s} is neither a directory nor a file")
    return path


def test_instance(instance: Instance, driver: Driver, temp_dir: Path) -> Status:
    """Execute a test on single instance."""

    # Clean temp directory
    for f in temp_dir.glob("*"):
        f.unlink()

    # Create temp instance
    temp_instance = Instance(
        model = temp_dir / "model.mzn",
        data = None if instance.data is None else temp_dir / "data.dzn",
    )

    # Create temp paths
    error_file = temp_dir / "error.txt"
    flatzinc_file = temp_dir / "instance.fzn"
    solution_file = temp_dir / "solution.dzn"
    check_file = temp_dir / "check.dzn"

    # Create symbolic links for temp instance
    for path, tmp_path in zip(instance.paths(), temp_instance.paths()):
        tmp_path.symlink_to(path.absolute())

    # Check instance is valid
    status, stderr = driver.check_instance(instance)
    if status != CmdStatus.OK:
        error_file.write_text(stderr)
        logging.debug(stderr)
        if status == CmdStatus.ERROR:
            return Status.INSTANCE_ERROR
        else:
            return Status.INSTANCE_TIMEOUT

    # Compile instance
    status, stderr = driver.compile(instance, args.solver, flatzinc_file)
    if status != CmdStatus.OK:
        error_file.write_text(stderr)
        logging.debug(stderr)
        if status == CmdStatus.ERROR:
            return Status.COMPILATION_ERROR
        else:
            return Status.COMPILATION_TIMEOUT

    # Solve instance
    status, stderr = driver.solve(instance, args.solver, solution_file)
    if status != CmdStatus.OK:
        error_file.write_text(stderr)
        logging.debug(stderr)
        if status == CmdStatus.ERROR:
            return Status.SOLVE_ERROR
        else:
            return Status.SOLVE_TIMEOUT

    # Check solution
    status, stderr = driver.check_solution(instance, args.check_solver, solution_file, check_file)
    if status != CmdStatus.OK:
        error_file.write_text(stderr)
        logging.debug(stderr)
        if status == CmdStatus.ERROR:
            return Status.CHECK_ERROR
        else:
            return Status.CHECK_TIMEOUT

    return Status.OK


def test_command(args: Namespace) -> None:
    """Execute test command."""
    SEPARATOR = "-"*40
    paths: list[Path] = args.input

    start_time = datetime.now()

    print(" ".join(sys.argv))
    print(SEPARATOR)

    # Create temp directory if needed
    temp_dir: Path = args.temp_dir
    temp_dir.mkdir(exist_ok=True)

    # Create minizinc driver
    driver = Driver(
        minzinc=args.minizinc,
        include_dir=args.include,
        timeout=args.timeout,
    )

    # Separate mzn files, dzn files and directories
    mzn_files = [p for p in paths if is_mzn(p)]
    dzn_files = [p for p in paths if is_dzn(p)]
    directories = [p for p in paths if p.is_dir()]

    # Collect all instances
    instances = Instance.combine(mzn_files, dzn_files)
    for directory in directories:
        dir_instances = Instance.collect(directory, recursive=args.recursive)
        instances.extend(dir_instances)
    instances.sort()

    status_count = {s:0 for s in Status}
    total_count = len(instances)

    # Run test on all instances
    for instance in instances:
        status = test_instance(instance, driver, temp_dir)
        status_count[status] += 1
        print(f"{status:^8}: {instance}")

    end_time = datetime.now()
    exec_duration = end_time - start_time

    # Time stats
    print(SEPARATOR)
    print(f"Start time: {start_time}")
    print(f"End time  : {end_time}")
    print(f"Duration  : {exec_duration}")

    print("-"*40)
    for status, count in status_count.items():
        percentage = count / total_count * 100
        print(f"{status:^8}: {count:>4}  {percentage:>4.1f}%")
    print(f"{'TOTAL':^8}: {total_count:>4}")

    # Exit with code 1 if any test is not OK
    if status_count[Status.OK] != total_count:
        sys.exit(1)


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="mznb",
        description="Minizinc test bench.",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--minizinc",
        default="minizinc",
        help="command to invoke minizinc",
        metavar="CMD",
    )

    parser.add_argument(
        "--temp-dir",
        default=Path("temp"),
        help="temporary directory",
        type=Path,
        metavar="DIR",
    )

    parser.add_argument(
        "-I", "--include",
        help="include directory",
        metavar="D",
    )

    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="verbose output (-vv for very verbose)",
    )

    parser.add_argument(
        "--solver",
        default="aries",
        help="solver to test (default: %(default)s)",
        metavar="S",
    )

    parser.add_argument(
        "--check-solver",
        default="gecode",
        help="solver to check (default: %(default)s)",
        metavar="S",
    )

    subparsers = parser.add_subparsers(
        help="command",
        metavar="cmd",
        required=True,
    )

    test_parser = subparsers.add_parser(
        name="test",
        help="test solver on instances",
        description="Test solver on minizinc instances.\n\n"
        "The inputs are treated as follows:\n"
        " - a directory gives on instance per mzn-dzn pair it contains\n"
        " - all mzn and dzn files are combined as if in one directory\n",
        formatter_class=RawTextHelpFormatter,
    )
    test_parser.set_defaults(fn=test_command)

    test_parser.add_argument(
        "--timeout",
        default=1.0,
        help="timeout in seconds (default: %(default)s)",
        type=float,
        metavar="T",
    )

    test_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="search instances recusively"
    )

    test_parser.add_argument(
        "input",
        help="directory or mzn/dzn file",
        nargs="+",
        type=input_mzn_dzn_dir,
    )

    # Parse command line args
    args = parser.parse_args()

    # Set log level depending of number of v:
    # default is WARNING
    # -v is INFO
    # -vv is DEBUG
    logging.basicConfig(level=30 - 10*args.v, format="%(message)s")

    args.fn(args)
