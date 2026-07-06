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
        # MeatLens Folder Inference Notebook
        ## Final 8-Sample CNN-Only Model, Seed 123

        This notebook loads the final 8-sample deployment CNN-only model for `seed=123`
        and runs folder inference on new images.

        Preprocessing before inference:
        - center-square ROI crop
        - resize to `224x224`
        - HSV/LAB threshold segmentation
        - neutral gray background fill
        - MobileNetV3Small preprocessing

        This notebook does not train. It only performs preprocessing + inference.
        """,
    )

    add_md(
        cells,
        """
        Important note:

        The model used here is the final deployment/demo model trained on all 8 validated samples.
        Official evaluation metrics should still be taken from the 8-fold cross-rotation experiment.
        """,
    )

    add_code(
        cells,
        """
        import io
        import json
        import shutil
        import zipfile
        from datetime import datetime
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from PIL import Image

        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenetv3

        from IPython.display import Image as DisplayImage, clear_output, display

        try:
            import ipywidgets as widgets
            IPYWIDGETS_AVAILABLE = True
        except Exception:
            widgets = None
            IPYWIDGETS_AVAILABLE = False

        import mobilenetv3small_segmented6_hybrid_lib as shared_lib
        from apply_hsv_lab_threshold_roi_batch import process_image

        print("TensorFlow:", tf.__version__)
        print("ipywidgets available:", IPYWIDGETS_AVAILABLE)
        print("SKIMAGE_AVAILABLE:", shared_lib.SKIMAGE_AVAILABLE)
        print("CV2_AVAILABLE:", shared_lib.CV2_AVAILABLE)
        """,
    )

    add_code(
        cells,
        """
        PROJECT_ROOT = Path.cwd()
        FINAL_ROOT = PROJECT_ROOT / "training_outputs" / "mobilenetv3small_8samples_final_deployment_cnn_only"
        MODELS_ROOT = FINAL_ROOT / "models"
        INFERENCE_ROOT = FINAL_ROOT / "inference_seed123"
        INFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

        MODEL_PATH = MODELS_ROOT / "meatlens_final_8samples_cnn_only_mobilenetv3small_seed123.h5"
        METADATA_PATH = MODELS_ROOT / "meatlens_final_8samples_cnn_only_mobilenetv3small_seed123_metadata.json"

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        LABEL_ORDER = list(metadata.get("label_order", ["fresh", "not fresh", "spoiled"]))
        INDEX_TO_LABEL = {
            int(k): v for k, v in metadata.get("index_to_label", {"0": "fresh", "1": "not fresh", "2": "spoiled"}).items()
        }
        INPUT_SHAPE = tuple(metadata.get("input_shape", [224, 224, 3]))
        MODEL_INPUT_MODE = str(metadata.get("model_input_mode", ""))
        IMAGE_CROP_MODE = str(metadata.get("image_crop_mode", ""))
        BACKGROUND_MODE = "gray"
        VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        if MODEL_INPUT_MODE != "cnn_only":
            raise ValueError(f"Unexpected model_input_mode: {MODEL_INPUT_MODE}")
        if IMAGE_CROP_MODE != "preprocessed_hsv_lab_threshold_roi_224":
            raise ValueError(f"Unexpected image_crop_mode: {IMAGE_CROP_MODE}")
        if INPUT_SHAPE != (224, 224, 3):
            raise ValueError(f"Unexpected input shape: {INPUT_SHAPE}")

        print("Model path:", MODEL_PATH)
        print("Metadata path:", METADATA_PATH)
        print("Label order:", LABEL_ORDER)
        print("Background mode:", BACKGROUND_MODE)
        """,
    )

    add_code(
        cells,
        """
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print("Model loaded successfully")
        print("Model input names:", model.input_names if hasattr(model, "input_names") else "unavailable")
        print("Model output shape:", model.output_shape)
        """,
    )

    add_code(
        cells,
        """
        def list_images_recursive(folder_path):
            folder = Path(folder_path)
            if not folder.exists() or not folder.is_dir():
                raise FileNotFoundError(f"Folder not found: {folder}")
            return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])


        def normalize_uploaded_items(upload_value):
            items = []

            if isinstance(upload_value, dict):
                for name, payload in upload_value.items():
                    content = payload.get("content") if isinstance(payload, dict) else None
                    if content is not None:
                        items.append({"name": name, "content": content})
                return items

            if isinstance(upload_value, (list, tuple)):
                for payload in upload_value:
                    if isinstance(payload, dict):
                        name = payload.get("name", "uploaded.zip")
                        content = payload.get("content")
                        if content is not None:
                            items.append({"name": name, "content": content})
                return items

            return items


        def extract_uploaded_zip_to_temp(upload_item):
            temp_root = INFERENCE_ROOT / "uploaded_zips"
            temp_root.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = temp_root / f"zip_{stamp}"
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            zip_path = work_dir / upload_item["name"]
            zip_path.write_bytes(upload_item["content"])

            extract_dir = work_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(upload_item["content"]), "r") as zf:
                zf.extractall(extract_dir)

            return extract_dir
        """,
    )

    add_code(
        cells,
        """
        def save_processed_image(image_uint8, out_path):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image_uint8).save(out_path, quality=95)
            return out_path


        def preview_processed_images(df, max_images=6):
            if df is None or len(df) == 0 or "processed_output_path" not in df.columns:
                print("No processed image previews available.")
                return

            preview_df = df.loc[df["processed_output_path"].astype(str).str.strip().ne("")].head(max_images).copy()
            if preview_df.empty:
                print("No saved processed images available for preview.")
                return

            for _, row in preview_df.iterrows():
                processed_path = Path(str(row["processed_output_path"]))
                if not processed_path.exists():
                    continue
                print(
                    f"{processed_path.name} | predicted={row.get('predicted_label', '')} | "
                    f"confidence={float(row.get('top_confidence', 0.0)):.4f}"
                )
                display(DisplayImage(filename=str(processed_path)))


        def predict_single_image(image_path, processed_output_path=None, background_mode=BACKGROUND_MODE):
            image_path = Path(image_path)
            segmented_uint8, seg_meta = process_image(image_path, background_mode=background_mode)

            if processed_output_path is not None:
                save_processed_image(segmented_uint8, processed_output_path)

            image_batch = np.expand_dims(
                preprocess_mobilenetv3(segmented_uint8.astype(np.float32).copy()),
                axis=0,
            ).astype(np.float32)

            probs = model.predict(image_batch, verbose=0)[0].astype(float)
            sorted_idx = np.argsort(-probs)
            top_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])

            top_class = INDEX_TO_LABEL[top_idx]
            second_class = INDEX_TO_LABEL[second_idx]
            top_confidence = float(probs[top_idx])
            second_confidence = float(probs[second_idx])
            prediction_margin = top_confidence - second_confidence

            transition_label, recommendation = shared_lib.transition_label_and_recommendation(
                top_class=top_class,
                top_confidence=top_confidence,
                second_class=second_class,
            )
            freshness_score = shared_lib.compute_freshness_score(
                prob_fresh=float(probs[0]),
                prob_not_fresh=float(probs[1]),
                prob_spoiled=float(probs[2]),
            )
            freshness_score_band = shared_lib.freshness_score_to_band(freshness_score)

            return {
                "image_path": str(image_path),
                "image_file_name": image_path.name,
                "processed_output_path": str(processed_output_path) if processed_output_path is not None else "",
                "predicted_label": top_class,
                "top_class": top_class,
                "top_confidence": top_confidence,
                "second_class": second_class,
                "second_confidence": second_confidence,
                "prediction_margin": float(prediction_margin),
                "prob_fresh": float(probs[0]),
                "prob_not_fresh": float(probs[1]),
                "prob_spoiled": float(probs[2]),
                "transition_label": transition_label,
                "recommendation": recommendation,
                "freshness_score": float(freshness_score),
                "freshness_score_band": freshness_score_band,
                "segmentation_failed": bool(seg_meta.get("segmentation_failed", False)),
                "mask_area_ratio": float(seg_meta.get("mask_area_ratio", 0.0)),
                "center_overlap_ratio": float(seg_meta.get("center_overlap_ratio", 0.0)),
                "number_of_components": int(seg_meta.get("number_of_components", 0)),
                "touches_border": bool(seg_meta.get("touches_border", False)),
                "background_mode": background_mode,
                "model_path": str(MODEL_PATH),
            }
        """,
    )

    add_code(
        cells,
        """
        def predict_folder(folder_path, save_csv=True, save_processed_images=True, background_mode=BACKGROUND_MODE):
            source_folder = Path(folder_path)
            image_paths = list_images_recursive(source_folder)
            if len(image_paths) == 0:
                raise ValueError(f"No supported image files found in: {source_folder}")

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = INFERENCE_ROOT / f"final_8samples_seed123_inference_{stamp}"
            processed_dir = run_dir / "processed_segmented_roi"
            run_dir.mkdir(parents=True, exist_ok=True)
            if save_processed_images:
                processed_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            errors = []

            for image_path in image_paths:
                try:
                    relative_path = image_path.relative_to(source_folder)
                    processed_output_path = processed_dir / relative_path if save_processed_images else None
                    rows.append(
                        predict_single_image(
                            image_path=image_path,
                            processed_output_path=processed_output_path,
                            background_mode=background_mode,
                        )
                    )
                except Exception as exc:
                    errors.append({"image_path": str(image_path), "error": repr(exc)})

            df = pd.DataFrame(rows)
            if len(df) > 0:
                df = df.sort_values(["freshness_score", "top_confidence"], ascending=[False, False]).reset_index(drop=True)

            summary = {
                "timestamp": stamp,
                "model_path": str(MODEL_PATH),
                "metadata_path": str(METADATA_PATH),
                "source_folder": str(source_folder),
                "run_dir": str(run_dir),
                "total_images_found": len(image_paths),
                "successful_predictions": int(len(df)),
                "errors": int(len(errors)),
                "background_mode": background_mode,
                "deployment_note": metadata.get("deployment_note", ""),
            }

            out_csv = None
            errors_csv = None
            summary_json = run_dir / "inference_summary.json"
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            if save_csv and len(df) > 0:
                out_csv = run_dir / "folder_predictions.csv"
                df.to_csv(out_csv, index=False)

            if len(errors) > 0:
                errors_csv = run_dir / "folder_inference_errors.csv"
                pd.DataFrame(errors).to_csv(errors_csv, index=False)

            return df, errors, summary, out_csv, errors_csv, run_dir
        """,
    )

    add_code(
        cells,
        """
        if IPYWIDGETS_AVAILABLE:
            upload_widget = widgets.FileUpload(
                accept=".zip",
                multiple=False,
                description="Upload ZIP",
            )
            folder_text = widgets.Text(
                value="",
                placeholder="Example: E:/Thesis Code/my_folder_of_images",
                description="Folder path:",
                layout=widgets.Layout(width="700px"),
            )
            run_button = widgets.Button(description="Run CNN-Only Inference", button_style="success")
            output_area = widgets.Output()

            display(widgets.HTML("<b>Option A:</b> Upload a ZIP file containing your folder of images."))
            display(upload_widget)
            display(widgets.HTML("<b>Option B:</b> Enter a local folder path and run."))
            display(folder_text)
            display(run_button)
            display(output_area)

            def _run_inference(_):
                with output_area:
                    clear_output(wait=True)
                    try:
                        folder_to_use = None
                        uploaded_items = normalize_uploaded_items(upload_widget.value)

                        if len(uploaded_items) > 0:
                            print("Using uploaded ZIP file...")
                            folder_to_use = extract_uploaded_zip_to_temp(uploaded_items[0])
                            print("Extracted to:", folder_to_use)
                        elif folder_text.value.strip():
                            folder_to_use = Path(folder_text.value.strip())
                            print("Using local folder:", folder_to_use)
                        else:
                            print("Please upload a ZIP file or enter a folder path.")
                            return

                        df, errors, summary, out_csv, errors_csv, run_dir = predict_folder(
                            folder_to_use,
                            save_csv=True,
                            save_processed_images=True,
                            background_mode=BACKGROUND_MODE,
                        )

                        print(json.dumps(summary, indent=2))
                        print("Run directory:", run_dir)
                        if out_csv is not None:
                            print("Predictions CSV:", out_csv)
                        if errors_csv is not None:
                            print("Errors CSV:", errors_csv)

                        if len(df) > 0:
                            display(df.head(30))
                            print("Processed image previews:")
                            preview_processed_images(df, max_images=6)
                        if len(errors) > 0:
                            display(pd.DataFrame(errors).head(20))
                    except Exception as exc:
                        print("Inference failed:", repr(exc))

            run_button.on_click(_run_inference)
        else:
            print("ipywidgets is not available. Use the manual mode cell below.")
        """,
    )

    add_code(
        cells,
        """
        # Manual mode
        # Set any folder containing source meat images, then run this cell.

        MANUAL_FOLDER_PATH = ""

        if MANUAL_FOLDER_PATH:
            df, errors, summary, out_csv, errors_csv, run_dir = predict_folder(
                MANUAL_FOLDER_PATH,
                save_csv=True,
                save_processed_images=True,
                background_mode=BACKGROUND_MODE,
            )
            print(json.dumps(summary, indent=2))
            print("Run directory:", run_dir)
            if out_csv is not None:
                print("Predictions CSV:", out_csv)
            if errors_csv is not None:
                print("Errors CSV:", errors_csv)
            display(df.head(30))
            print("Processed image previews:")
            preview_processed_images(df, max_images=6)
            if len(errors) > 0:
                display(pd.DataFrame(errors).head(20))
        else:
            print("Set MANUAL_FOLDER_PATH first.")
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
    output_path = Path("new9_folder_inference_final_8samples_cnn_only_seed123.ipynb")
    notebook = build_notebook()
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {output_path.resolve()}")


if __name__ == "__main__":
    main()
