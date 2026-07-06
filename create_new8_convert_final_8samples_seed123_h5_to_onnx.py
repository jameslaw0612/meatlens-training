#!/usr/bin/env python3
import json
import uuid
from pathlib import Path
from textwrap import dedent


def add_md(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def add_code(cells: list[dict], text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
        }
    )


def build_notebook() -> dict:
    cells: list[dict] = []

    add_md(
        cells,
        """
        # Convert Final 8-Sample Seed123 H5 To ONNX

        This notebook converts the final deployment H5 model for `seed=123` into ONNX.

        Target model:
        - `training_outputs/mobilenetv3small_8samples_final_deployment_cnn_only/models/meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.h5`

        Output:
        - `training_outputs/mobilenetv3small_8samples_final_deployment_cnn_only/models/meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.onnx`
        """,
    )

    add_code(
        cells,
        """
        from pathlib import Path
        import json
        import importlib

        PROJECT_ROOT = Path.cwd()
        MODELS_ROOT = PROJECT_ROOT / "training_outputs" / "mobilenetv3small_8samples_final_deployment_cnn_only" / "models"

        MODEL_H5_PATH = MODELS_ROOT / "meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.h5"
        MODEL_ONNX_PATH = MODELS_ROOT / "meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.onnx"
        MODEL_METADATA_PATH = MODELS_ROOT / "meatlens_final_8samples_cnn_only_mobilenetv3small_seed123_metadata.json"

        print("MODEL_H5_PATH =", MODEL_H5_PATH)
        print("MODEL_ONNX_PATH =", MODEL_ONNX_PATH)
        print("MODEL_METADATA_PATH =", MODEL_METADATA_PATH)
        print("H5 exists?", MODEL_H5_PATH.exists())
        print("Metadata exists?", MODEL_METADATA_PATH.exists())
        """,
    )

    add_code(
        cells,
        """
        import importlib.util

        def package_available(module_name: str) -> bool:
            return importlib.util.find_spec(module_name) is not None

        print("tensorflow available?", package_available("tensorflow"))
        print("tf2onnx available?", package_available("tf2onnx"))
        print("onnx available?", package_available("onnx"))
        """,
    )

    add_code(
        cells,
        """
        import tensorflow as tf
        import tf2onnx

        try:
            import onnx
        except Exception:
            onnx = None

        def convert_seed123_h5_to_onnx(
            model_h5_path=MODEL_H5_PATH,
            model_onnx_path=MODEL_ONNX_PATH,
            opset=13,
        ):
            if not model_h5_path.exists():
                raise FileNotFoundError(f"Missing H5 model: {model_h5_path}")

            model = tf.keras.models.load_model(model_h5_path, compile=False)

            input_signature = (
                tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image_input"),
            )

            model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=opset,
                output_path=str(model_onnx_path),
            )

            result = {
                "model_h5_path": str(model_h5_path),
                "model_onnx_path": str(model_onnx_path),
                "opset": opset,
                "onnx_exists": model_onnx_path.exists(),
                "external_tensor_storage": external_tensor_storage,
            }

            if model_onnx_path.exists():
                result["onnx_size_mb"] = model_onnx_path.stat().st_size / (1024 * 1024)

            if onnx is not None and model_onnx_path.exists():
                onnx_model = onnx.load(str(model_onnx_path))
                onnx.checker.check_model(onnx_model)
                result["onnx_inputs"] = [node.name for node in onnx_model.graph.input]
                result["onnx_outputs"] = [node.name for node in onnx_model.graph.output]

            return result
        """,
    )

    add_code(
        cells,
        """
        if MODEL_METADATA_PATH.exists():
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            metadata
        else:
            print("Metadata JSON not found.")
        """,
    )

    add_code(
        cells,
        """
        MANUAL_CONFIRM_CONVERT_TO_ONNX = False

        if MANUAL_CONFIRM_CONVERT_TO_ONNX:
            onnx_result = convert_seed123_h5_to_onnx()
            onnx_result
        else:
            print("Seed123 H5 to ONNX conversion is ready but not started.")
            print("Set MANUAL_CONFIRM_CONVERT_TO_ONNX = True to convert.")
        """,
    )

    add_md(
        cells,
        """
        After conversion, the ONNX file should be here:

        - `training_outputs/mobilenetv3small_8samples_final_deployment_cnn_only/models/meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.onnx`
        """,
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output_path = Path("new8_convert_final_8samples_seed123_h5_to_onnx.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
