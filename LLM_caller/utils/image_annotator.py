"""
在 GUI 截图上标注点击位置和 UI 元素包围框，供多模态模型使用。
"""
import os
from PIL import Image, ImageDraw


class ImageAnnotator:
    """
    在截图上绘制：
    - 红色十字准星 + 圆圈 — 点击位置
    - 绿色矩形框 — UI 元素边界
    """

    def __init__(self, crosshair_size: int = 28, line_width: int = 4):
        self.crosshair_size = crosshair_size
        self.line_width = line_width

    def annotate_step(
        self,
        image_path: str,
        action_x: int | None,
        action_y: int | None,
        element_bbox: list | None = None,
        output_dir: str = ".",
        tag: str = "annotated",
    ) -> str | None:
        """
        标注单张图片并保存。

        - 有点击坐标时绘制红色十字准星 + 绿色元素框
        - 无坐标时（如 Back 操作）直接复制原图

        Returns:
            标注后图片路径，或 None（图片不存在）
        """
        if not image_path or not os.path.exists(image_path):
            return None

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        red = (255, 42, 42)
        green = (46, 204, 113)
        sz = self.crosshair_size
        lw = self.line_width

        if action_x is not None and action_y is not None:
            draw = ImageDraw.Draw(img)

            # 点击坐标裁剪到图片范围内
            cx = max(0, min(action_x, w - 1))
            cy = max(0, min(action_y, h - 1))

            # ── 十字准星 ──
            draw.line([(cx - sz, cy), (cx + sz, cy)], fill=red, width=lw)
            draw.line([(cx, cy - sz), (cx, cy + sz)], fill=red, width=lw)

            # ── 外圈（双圆强调）──
            for r in (sz + 6, sz - 4):
                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    outline=red, width=lw // 2,
                )

            # ── 元素包围框 ──
            if element_bbox and len(element_bbox) == 4:
                left = max(0, min(element_bbox[0], w - 1))
                top = max(0, min(element_bbox[1], h - 1))
                right = max(0, min(element_bbox[2], w - 1))
                bottom = max(0, min(element_bbox[3], h - 1))
                if right > left and bottom > top:
                    draw.rectangle(
                        [left, top, right, bottom],
                        outline=green, width=lw,
                    )

        # ── 缩小尺寸（长边不超过 768px）加速传输 ──
        max_dim = 768
        if w > max_dim or h > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # ── 保存 ──
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        stem, _ = os.path.splitext(basename)
        out_path = os.path.join(output_dir, f"{stem}_{tag}.jpg")
        img.save(out_path, "JPEG", quality=75)
        return out_path
