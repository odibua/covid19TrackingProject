import distutils.cmd
import itertools
# TODO(odibua@): Investigate potential security issues here when
# intepretML more developed
import subprocess
from setuptools import setup, find_packages

# TODO(odibua@): Add filter based on config file


def get_python_files():
    get_py_files_command = ["git", "ls-files", "|", "grep", "*.py"]
    py_files = subprocess.check_output(get_py_files_command).split(b"\n")[0:-1]
    py_files = [py_file for py_file in py_files if py_file !=
                b"sensitivity_methods/SALib/sample/directions.py"]
    return py_files


class AutoPep8Command(distutils.cmd.Command):
    """A custom command to run autopep8 to fix linting issues"""
    description = 'Run autopep8 on all .py files tracked by git'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        """Run autopep8 on all files tracked by git that end in .py."""
        self.announce("Running autopep8")
        # Get all python files tracked by git
        py_files = get_python_files()

        # Run autopep8 on tracked python files
        autopep8_command = [
            "autopep8",
            "--in-place",
            "--aggressive",
            "--max-line-length",
            "120",
            *py_files]
        subprocess.check_call(autopep8_command)


class PyCodeStyleCommand(distutils.cmd.Command):
    """A custom command to run pycode style on all py files being tracked by git"""
    description = 'Check style of git tracked python files using pycodestyle'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        """Run pycodestyle on all files tracked by git that end in .py."""
        self.announce("Running pycodestyle")
        # Get all python files tracked by git
        # Get all python files tracked by git
        py_files = get_python_files()

        # Run pycodestyle on tracked python files
        pycodestyle_command = ["pycodestyle", "--ignore=W503,W504", *py_files]
        pycodestyle_command_process = subprocess.Popen(
            pycodestyle_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pycodestyle_out, _ = pycodestyle_command_process.communicate()

        # Raise exceptions if style issues found
        if len(pycodestyle_out) > 0:
            print(pycodestyle_out.decode("utf-8"))
            raise Exception(
                f"Style errors found. Fix by running python setup.py pylint and manually change remaining errors after"
                f"rerunning python setup.py checkpycodestyle")


setup(
    name='covid19Tracking',
    version='0.2.0',
    packages=find_packages(
        include=[
            'states',
            'states.*']),
    install_requires=[
        'autopep8',
        'bokeh',
        'celery',
        'beautifulsoup4',
        'html5lib',
        'numpy>=1.9.0',
        'pandas',
        'requests',
        'requests_html',
        'scipy',
        'setuptools>=40.0',
        "typing",
        'urllib3>1.25',
        'wheel',
        'PyYAML',
    ],
    cmdclass={
        'checkpycodestyle': PyCodeStyleCommand,
        'pylint': AutoPep8Command,
    },
    extras_require={
        'interactive': [
            'matplotlib>=2.2.0']},
    setup_requires=[
        'pycodestyle',
        'pytest-runner'],
    tests_require=[
        'pre-commit',
        'pytest',
        'pytest-cov',
        'recommonmark',
    ],
)
