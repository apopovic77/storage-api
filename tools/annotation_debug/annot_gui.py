"""
Interactive Tkinter GUI for managing storage object annotations.
Allows loading images, triggering AI analysis, and visualising annotation
overlays in one place.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk


MAX_DISPLAY_WIDTH = 900
POLL_INTERVAL_SECONDS = 2.5


@dataclass
class Annotation:
    label: str
    kind: str
    anchor_x: float
    anchor_y: float
    box: Optional[List[float]]
    confidence: Optional[float]


class AnnotationGUI:
    def __init__(self, api_key: str, base_url: str):
        self.default_api_key = api_key
        self.default_base_url = base_url.rstrip("/")
        self.product_metadata: Optional[Dict[str, Any]] = None
        self.product_api_base = "https://api.oneal.eu/v1"

        self.root = tk.Tk()
        self.root.title("Storage Annotation GUI")

        self._build_layout()

        self.current_object: Optional[Dict[str, Any]] = None
        self.current_annotations: List[Annotation] = []
        self.current_prompt: str = ""
        self.current_response: str = ""

        self.pil_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.display_scale = 1.0

    # ------------------------------------------------------------------ UI
    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill=tk.X)

        ttk.Label(header, text="Object ID").grid(row=0, column=0, sticky=tk.W)
        self.object_entry = ttk.Entry(header, width=16)
        self.object_entry.grid(row=0, column=1, padx=(5, 15))

        ttk.Label(header, text="API Key").grid(row=0, column=2, sticky=tk.W)
        self.api_key_entry = ttk.Entry(header, width=36)
        self.api_key_entry.grid(row=0, column=3, padx=(5, 15))

        ttk.Label(header, text="Base URL").grid(row=0, column=4, sticky=tk.W)
        self.base_entry = ttk.Entry(header, width=36)
        self.base_entry.grid(row=0, column=5)

        self.api_key_entry.insert(0, self.default_api_key or "")
        self.base_entry.insert(0, self.default_base_url)

        product_bar = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        product_bar.pack(fill=tk.X)

        ttk.Label(product_bar, text="Product ID").grid(row=0, column=0, sticky=tk.W)
        self.product_id_entry = ttk.Entry(product_bar, width=18)
        self.product_id_entry.grid(row=0, column=1, padx=(5, 12))

        ttk.Label(product_bar, text="Product API Base").grid(row=0, column=2, sticky=tk.W)
        self.product_base_entry = ttk.Entry(product_bar, width=34)
        self.product_base_entry.grid(row=0, column=3, padx=(5, 12))
        self.product_base_entry.insert(0, self.product_api_base)

        ttk.Label(product_bar, text="Product API Key").grid(row=0, column=4, sticky=tk.W)
        self.product_api_key_entry = ttk.Entry(product_bar, width=32)
        self.product_api_key_entry.grid(row=0, column=5, padx=(5, 12))
        self.product_api_key_entry.insert(0, self.default_api_key or "")

        ttk.Button(product_bar, text="Load Product Specs", command=self.on_load_product_specs).grid(row=0, column=6)

        buttons = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        buttons.pack(fill=tk.X)

        ttk.Button(buttons, text="Load Object", command=self.on_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons, text="Analyze (quality)", command=self.on_analyze).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons, text="Fetch annotations", command=self.on_fetch_annotations).pack(side=tk.LEFT, padx=5)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, width=640, height=640, bg="#222222")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=360)
        side.pack(side=tk.RIGHT, fill=tk.BOTH)

        ttk.Label(side, text="Product Metadata (context)").pack(anchor=tk.W)
        self.metadata_box = ScrolledText(side, width=64, height=10)
        self.metadata_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        ttk.Label(side, text="Annotations").pack(anchor=tk.W)
        self.annotations_box = ScrolledText(side, width=64, height=12)
        self.annotations_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        ttk.Label(side, text="Prompt").pack(anchor=tk.W)
        self.prompt_box = ScrolledText(side, width=64, height=10)
        self.prompt_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        ttk.Label(side, text="AI Response").pack(anchor=tk.W)
        self.response_box = ScrolledText(side, width=64, height=10)
        self.response_box.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        ttk.Label(side, text="Log").pack(anchor=tk.W)
        self.log_box = ScrolledText(side, width=64, height=8)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- helpers
    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)

    def _headers(self) -> Dict[str, str]:
        key = self.api_key_entry.get().strip() or self.default_api_key
        headers: Dict[str, str] = {}
        if key:
            headers["X-API-KEY"] = key
        return headers

    def _base_url(self) -> str:
        url = self.base_entry.get().strip() or self.default_base_url
        return url.rstrip("/")

    def _get_object_id(self) -> Optional[int]:
        raw = self.object_entry.get().strip()
        if not raw:
            messagebox.showerror("Fehlende ID", "Bitte eine Storage Object ID eingeben.")
            return None
        try:
            return int(raw)
        except ValueError:
            messagebox.showerror("Ungültige ID", f"'{raw}' ist keine gültige Zahl.")
            return None

    def _fetch_json(self, path: str) -> Dict[str, Any]:
        url = f"{self._base_url()}{path}" if path.startswith("/") else path
        resp = requests.get(url, headers=self._headers(), timeout=40)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp)
        return resp.json()

    def _post_json(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self._base_url()}{path}" if path.startswith("/") else path
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp)
        return resp.json()

    def _fetch_image(self, object_id: int) -> Image.Image:
        base_url = self._base_url()
        specs = [
            "trim=true&width=1400&format=webp&quality=80",
            "trim=true&width=1200&format=jpeg&quality=85",
            "trim=true&width=900&format=jpeg&quality=85",
            "trim=true&width=720&format=jpeg&quality=80",
            "trim=true",
        ]

        last_exc: Optional[Exception] = None

        def _download(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
            resp = requests.get(url, headers=headers or {}, timeout=(20, 180), stream=True)
            resp.raise_for_status()
            chunks: List[bytes] = []
            for chunk in resp.iter_content(65536):
                if chunk:
                    chunks.append(chunk)
            data = b"".join(chunks)
            if not data:
                raise RuntimeError("empty image response")
            return data

        for spec in specs:
            url = f"{base_url}/storage/media/{object_id}?{spec}&_={int(time.time())}"
            try:
                data = _download(url, headers=self._headers())
                return Image.open(BytesIO(data)).convert("RGBA")
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                self.log(f"Bild-Download fehlgeschlagen ({spec}): {exc}")

        # Try external fallbacks using object metadata (file_url / external_uri)
        try:
            obj = self.current_object or self._fetch_json(f"/storage/objects/{object_id}")
        except Exception as exc:  # pylint: disable=broad-except
            obj = None
            last_exc = exc

        if obj:
            candidates = []
            for key in ("webview_url", "file_url", "external_uri"):
                value = obj.get(key)
                if value:
                    candidates.append((key, value))

            for label, url in candidates:
                try:
                    data = _download(url)
                    self.log(f"Bild über {label} geladen.")
                    return Image.open(BytesIO(data)).convert("RGBA")
                except Exception as exc:  # pylint: disable=broad-except
                    last_exc = exc
                    self.log(f"Fallback {label} fehlgeschlagen: {exc}")

        raise RuntimeError("Bild konnte nicht geladen werden") from last_exc

    # ----------------------------------------------------------- operations
    def on_load(self) -> None:
        object_id = self._get_object_id()
        if object_id is None:
            return

        def worker():
            try:
                self.log(f"Lade Objekt {object_id}…")
                data = self._fetch_json(f"/storage/objects/{object_id}")
                self.current_object = data
                self.current_prompt = (data.get("ai_context_metadata") or {}).get("prompt", "")
                self.current_response = (data.get("ai_context_metadata") or {}).get("response", "")

                annotations = self._annotations_from_object(data)
                self.current_annotations = annotations

                image = self._fetch_image(object_id)
                self._update_ui(image, annotations)
            except Exception as exc:  # pylint: disable=broad-except
                self._handle_error("Fehler beim Laden", exc)

        threading.Thread(target=worker, daemon=True).start()

    def on_fetch_annotations(self) -> None:
        object_id = self._get_object_id()
        if object_id is None:
            return

        def worker():
            try:
                data = self._fetch_json(f"/storage/objects/{object_id}/annotations")
                annotations = self._parse_annotations(data.get("annotations", []))
                self.current_annotations = annotations
                self.log(f"{len(annotations)} Annotationen geladen.")
                self._refresh_canvas()
                self._update_annotation_list()
            except Exception as exc:  # pylint: disable=broad-except
                self._handle_error("Annotation Fetch fehlgeschlagen", exc)

        threading.Thread(target=worker, daemon=True).start()

    def on_analyze(self) -> None:
        object_id = self._get_object_id()
        if object_id is None:
            return

        metadata_payload = self._resolve_metadata_context()

        def worker():
            try:
                self.log("Starte Analyse über /analyze-async…")
                route = (
                    f"/storage/analyze-async/{object_id}?"
                    "mode=quality&ai_tasks=vision,embedding,kg&"
                    "ai_vision_mode=product&ai_context_role=product"
                )
                if metadata_payload is not None:
                    try:
                        import json as _json
                        from urllib.parse import quote_plus

                        meta_json = _json.dumps(metadata_payload, ensure_ascii=False)
                        route += f"&ai_metadata={quote_plus(meta_json)}"
                    except Exception as exc:  # pylint: disable=broad-except
                        self.log(f"Konnte ai_metadata nicht serialisieren: {exc}")

                result = self._post_json(route)
                task_id = result.get("task_id")
                if not task_id:
                    raise RuntimeError(f"Keine task_id im Response: {result}")

                while True:
                    info = self._fetch_json(f"/storage/tasks/{task_id}")
                    status = info.get("status")
                    progress = info.get("progress")
                    phase = info.get("current_phase")
                    self.log(f"Task {task_id}: {status} ({progress}% – {phase})")
                    if status in {"completed", "failed"}:
                        break
                    time.sleep(POLL_INTERVAL_SECONDS)

                if status == "failed":
                    raise RuntimeError(f"Analyse fehlgeschlagen: {info.get('error')}")

                self.log("Analyse erfolgreich – lade Daten neu…")
                data = self._fetch_json(f"/storage/objects/{object_id}")
                self.current_object = data
                self.current_prompt = (data.get("ai_context_metadata") or {}).get("prompt", "")
                self.current_response = (data.get("ai_context_metadata") or {}).get("response", "")
                annotations = self._annotations_from_object(data)
                self.current_annotations = annotations
                image = self._fetch_image(object_id)
                self._update_ui(image, annotations)
            except Exception as exc:  # pylint: disable=broad-except
                self._handle_error("Analyse fehlgeschlagen", exc)

        threading.Thread(target=worker, daemon=True).start()

    def on_load_product_specs(self) -> None:
        product_id = self.product_id_entry.get().strip()
        if not product_id:
            messagebox.showerror("Fehlende Produkt-ID", "Bitte eine Product ID eingeben.")
            return

        base = self.product_base_entry.get().strip() or self.product_api_base
        key = self.product_api_key_entry.get().strip() or self.default_api_key

        def worker():
            try:
                url = f"{base.rstrip('/')}/products/{product_id}?format=resolved"
                headers = {"X-API-Key": key} if key else {}
                resp = requests.get(url, headers=headers, timeout=45)
                resp.raise_for_status()
                product = resp.json()
                metadata = self._build_product_metadata(product)
                self.product_metadata = metadata
                import json as _json

                self.metadata_box.delete("1.0", tk.END)
                self.metadata_box.insert(tk.END, _json.dumps(metadata, ensure_ascii=False, indent=2))
                specs_count = len(metadata.get("specifications", []))
                features_count = len(metadata.get("features", []))
                self.log(f"Produkt {product_id} geladen – {specs_count} Spezifikationen, {features_count} Features.")
            except Exception as exc:  # pylint: disable=broad-except
                self._handle_error("Product Specs laden fehlgeschlagen", exc)

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------------------------------------- annotation
    def _annotations_from_object(self, obj: Dict[str, Any]) -> List[Annotation]:
        meta = obj.get("ai_context_metadata") or {}
        embedding_annotations = meta.get("embedding_info", {}).get("metadata", {}).get("annotations")
        if embedding_annotations:
            return self._parse_annotations(embedding_annotations)

        stored = meta.get("annotations") or obj.get("annotations")
        if stored:
            return self._parse_annotations(stored)

        try:
            data = self._fetch_json(f"/storage/objects/{obj['id']}/annotations")
            return self._parse_annotations(data.get("annotations", []))
        except Exception:  # pylint: disable=broad-except
            return []

    def _parse_annotations(self, raw_items: List[Any]) -> List[Annotation]:
        annotations: List[Annotation] = []
        for item in raw_items:
            if not item:
                continue
            anchor = item.get("anchor") or {}
            annotations.append(
                Annotation(
                    label=item.get("label") or "(no label)",
                    kind=item.get("type") or "unknown",
                    anchor_x=float(anchor.get("x", 0.0)),
                    anchor_y=float(anchor.get("y", 0.0)),
                    box=item.get("box"),
                    confidence=item.get("confidence"),
                )
            )
        return annotations

    # ------------------------------------------------------------ rendering
    def _update_ui(self, image: Image.Image, annotations: List[Annotation]) -> None:
        self.pil_image = image
        self._refresh_canvas()
        self._update_annotation_list()

        self.prompt_box.delete("1.0", tk.END)
        self.prompt_box.insert(tk.END, self.current_prompt or "")

        self.response_box.delete("1.0", tk.END)
        self.response_box.insert(tk.END, self.current_response or "")

        self.log(f"Bild {image.width}x{image.height}px, {len(annotations)} Annotationen")

    def _refresh_canvas(self) -> None:
        if not self.pil_image:
            return

        width, height = self.pil_image.size
        scale = min(1.0, MAX_DISPLAY_WIDTH / float(width))
        self.display_scale = scale

        if scale < 1.0:
            display_image = self.pil_image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        else:
            display_image = self.pil_image

        self.tk_image = ImageTk.PhotoImage(display_image)
        disp_width, disp_height = display_image.size

        self.canvas.configure(width=disp_width, height=disp_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for ann in self.current_annotations:
            self._draw_annotation(ann, disp_width, disp_height)

    def _draw_annotation(self, ann: Annotation, disp_width: int, disp_height: int) -> None:
        x = max(0.0, min(1.0, ann.anchor_x)) * disp_width
        y = max(0.0, min(1.0, ann.anchor_y)) * disp_height

        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#33ff00", outline="")
        self.canvas.create_text(x + 8, y, text=ann.label, anchor=tk.W, fill="#ffffff", font=("Arial", 10, "bold"))

        if ann.box and len(ann.box) == 4:
            x1, y1, x2, y2 = self._normalize_box(ann.box)
            self.canvas.create_rectangle(x1 * disp_width, y1 * disp_height, x2 * disp_width, y2 * disp_height,
                                         outline="#00d6ff", width=2)

    def _normalize_box(self, box: List[float]) -> Tuple[float, float, float, float]:
        if len(box) != 4:
            return (0.0, 0.0, 0.0, 0.0)

        x1, y1, x2, y2 = box
        # Determine whether box is [x, y, width, height]
        if x2 <= 1.0 and y2 <= 1.0 and x2 > x1 and y2 > y1:
            pass  # already x1/x2 format
        else:
            x2 = x1 + max(0.0, min(1.0, x2))
            y2 = y1 + max(0.0, min(1.0, y2))

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        if x2 - x1 < 0.01:
            x1, x2 = 0.0, 1.0
        if y2 - y1 < 0.01:
            y1, y2 = 0.0, 1.0

        return x1, y1, x2, y2

    def _resolve_metadata_context(self) -> Optional[Dict[str, Any]]:
        raw = self.metadata_box.get("1.0", tk.END).strip()
        if raw:
            try:
                import json as _json

                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:  # pylint: disable=broad-except
                self.log(f"Metadata JSON ungültig: {exc}")
        return self.product_metadata

    def _build_product_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "product": {
                "id": product.get("id"),
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category_ids": product.get("category_ids"),
                "sku": product.get("sku") or product.get("slug"),
            }
        }

        specs: List[Dict[str, Any]] = []
        for item in (product.get("specifications") or []):
            label = item.get("label") or item.get("name")
            value = item.get("value")
            if not label or value is None:
                continue
            entry: Dict[str, Any] = {"label": label, "value": value}
            unit = item.get("unit")
            if unit:
                entry["unit"] = unit
            specs.append(entry)
        if specs:
            metadata["specifications"] = specs

        features: List[Dict[str, Any]] = []
        for source in (product.get("technical_details") or []):
            if isinstance(source, str):
                features.append({"label": source})
            elif isinstance(source, dict):
                label = source.get("label") or source.get("title")
                description = source.get("description") or source.get("text")
                if label:
                    entry = {"label": label}
                    if description:
                        entry["description"] = description
                    features.append(entry)
        for source in (product.get("features") or []):
            if isinstance(source, str):
                features.append({"label": source})
            elif isinstance(source, dict):
                label = source.get("label") or source.get("name")
                if not label:
                    continue
                entry = {"label": label}
                description = source.get("description")
                if description:
                    entry["description"] = description
                features.append(entry)
        if features:
            metadata["features"] = features

        expected = [item.get("label") for item in features if item.get("label")]
        metadata["vision_objectives"] = {
            "instructions": "Locate and describe each listed product feature or specification within the image using precise pixel anchors.",
            "expected_outputs": expected[:20],
        }

        return metadata

    def _update_annotation_list(self) -> None:
        self.annotations_box.delete("1.0", tk.END)
        for ann in self.current_annotations:
            conf = f" ({ann.confidence:.2%})" if ann.confidence is not None else ""
            self.annotations_box.insert(tk.END, f"- {ann.label} [{ann.kind}]{conf}\n")

    # -------------------------------------------------------------- errors
    def _handle_error(self, title: str, exc: Exception) -> None:
        self.log(f"{title}: {exc}")
        messagebox.showerror(title, str(exc))

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI für Storage Annotation Tests")
    parser.add_argument("--api-key", required=True, help="Storage API Key")
    parser.add_argument("--base-url", default="https://api-storage.arkturian.com", help="Storage API Base URL")
    args = parser.parse_args()

    gui = AnnotationGUI(api_key=args.api_key, base_url=args.base_url)
    gui.run()


if __name__ == "__main__":
    main()

