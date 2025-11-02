import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1) 必要：head（第三人称）视角
        if "observation/image" not in data:
            raise KeyError("Missing key: 'observation/image' (head camera).")
        base_image = _parse_image(data["observation/image"])

        # 2) 可选：left_wrist（左腕）与 right_wrist（右腕）
        #    - 若存在，对应 mask=True
        #    - 若不存在，用 zeros_like(base) 占位，但 mask=False（不参与模型计算）
        if "observation/left_wrist_image" in data:
            # ✅ 优先支持 "left_wrist_image"（你当前数据集的命名）
            left_wrist = _parse_image(data["observation/left_wrist_image"])
            left_mask = np.True_
        elif "observation/wrist_image" in data:
            # 兼容旧版 Libero 命名
            left_wrist = _parse_image(data["observation/wrist_image"])
            left_mask = np.True_
        else:
            # 若都不存在，则填充全零并设为 mask=False
            left_wrist = np.zeros_like(base_image)
            left_mask = np.False_

        # 右腕的键名给两种常见写法都兼容一下：
        right_key = (
            "observation/right_wrist_image"
            if "observation/right_wrist_image" in data
            else ("observation/wrist_image_right" if "observation/wrist_image_right" in data else None)
        )
        if right_key is not None:
            right_wrist = _parse_image(data[right_key])
            right_mask = np.True_
        else:
            right_wrist = np.zeros_like(base_image)
            # NOTE: 原仓库里只在 PI0-FAST 时对 padding 置 True；这里统一按“缺失即不参与”处理
            right_mask = np.False_

        # 3) 组装模型期望的输入结构（键名保持不变）
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # 4) 训练监督：动作序列（若存在）
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 5) 任务文本（若存在）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs



@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
