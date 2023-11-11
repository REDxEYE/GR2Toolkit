def plugin_init():
    pass


def gr2_init():
    pass


def gr2_load(operator, filepath: str, files: list[str]):
    pass


plugin_info = {
    "name": "GR2(granny2) model importer",
    "id": "GR2Toolkit",
    "description": "Plugin to load .gr2 files",
    "version": (0, 1, 0),
    "loaders": [
        {
            "name": "Load .gr2 file",
            "id": "gr2",
            "exts": ("*.gr2",),
            "init_fn": gr2_init,
            "import_fn": gr2_load,
            "properties": [

            ]
        },
    ],
    "init_fn": plugin_init
}
