import os
import shutil
import tempfile
import atexit


def load_pipeline_scalers_for_side(pipeline, model_path, side: str):
    """
    Carga los scalers correctos para un side (long/short).

    Prioridad:
        1. scalers_<release>_<side>/
        2. scalers_<release>/
    """

    model_dir = os.path.dirname(model_path)
    release = pipeline.general_config.release

    side_dir = os.path.join(model_dir, f"scalers_{release}_{side}")
    generic_dir = os.path.join(model_dir, f"scalers_{release}")

    if os.path.isdir(side_dir):

        # crear directorio temporal
        tmp_root = tempfile.mkdtemp(prefix=f"scalers_{side}_")

        # registrar limpieza automática al salir el proceso
        atexit.register(lambda: shutil.rmtree(tmp_root, ignore_errors=True))

        shutil.copytree(
            side_dir,
            os.path.join(tmp_root, f"scalers_{release}")
        )

        pipeline.load_scalers(base_path=tmp_root)

    elif os.path.isdir(generic_dir):

        pipeline.load_scalers(base_path=model_dir)

    else:

        raise RuntimeError(
            f"No scalers found for release={release} side={side} in {model_dir}"
        )