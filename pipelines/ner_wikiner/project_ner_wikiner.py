from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


def wikiner_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all")


if __name__ == '__main__':
    wikiner_project()
