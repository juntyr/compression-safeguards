import importlib.metadata
import tomllib


def define_env(env):
    @env.macro
    def version(x):
        return importlib.metadata.version(x)

    @env.macro
    def pyproject():
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        return pyproject
