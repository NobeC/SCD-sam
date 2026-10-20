import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage import measure
except ImportError:
    measure = None


# =========================================================
# 1. 参数
# =========================================================
NPZ_PATH = "/home/homenew/yang/SCD/SCD-SAM/SCD/sam_mask_cache/Landsat/train/From2008To2020_00.npz"
OUTPUT_DIR = "SCD-SAM/SCD/sam_mask_visualization"

POINTS_PER_SIDE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "A_masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "B_masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "A_mask_with_grid"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "B_mask_with_grid"), exist_ok=True)

# =========================================================
# 2. 读取 npz
# =========================================================
data = np.load(NPZ_PATH, allow_pickle=True)

print("Available keys:")
for k in data.files:
    arr = data[k]
    print(f"{k:25s} shape={arr.shape}, dtype={arr.dtype}")

H, W = map(int, data["image_shape"])
print(f"\nImage shape: H={H}, W={W}")

# =========================================================
# 3. 解压 packed mask
# =========================================================
def unpack_masks(packed, width):
    """
    packed shape:
        [N, H, ceil(W/8)]

    output:
        [N, H, W], bool
    """
    masks = np.unpackbits(
        packed,
        axis=-1
    )

    masks = masks[:, :, :width]

    return masks.astype(bool)


masks_A = unpack_masks(data["packed_A"], W)
masks_B = unpack_masks(data["packed_B"], W)

print("\nUnpacked masks:")
print("A:", masks_A.shape)
print("B:", masks_B.shape)

# =========================================================
# 4. 保存每个 mask 为黑白 PNG
# =========================================================
def save_individual_masks(masks, prefix, out_dir):
    for i, mask in enumerate(masks):
        save_path = os.path.join(
            out_dir,
            f"{prefix}_mask_{i:03d}.png"
        )

        plt.imsave(
            save_path,
            mask.astype(np.uint8),
            cmap="gray",
            vmin=0,
            vmax=1
        )


save_individual_masks(
    masks_A,
    "A",
    os.path.join(OUTPUT_DIR, "A_masks")
)

save_individual_masks(
    masks_B,
    "B",
    os.path.join(OUTPUT_DIR, "B_masks")
)

print("\nIndividual masks saved.")

# =========================================================
# 5. 生成 32×32 基础 prompt grid
# =========================================================
def create_prompt_grid(H, W, points_per_side=32):
    """
    基础 AMG prompt grid。
    注意：
    这是基础层近似网格位置。
    若 crop_n_layers > 0，
    实际 SAM 使用的 prompt 点还会更多。
    """

    xs = (
        np.arange(points_per_side) + 0.5
    ) * W / points_per_side

    ys = (
        np.arange(points_per_side) + 0.5
    ) * H / points_per_side

    xx, yy = np.meshgrid(xs, ys)

    points = np.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        axis=1
    )

    return points


prompt_grid = create_prompt_grid(
    H,
    W,
    POINTS_PER_SIDE
)

print(
    f"Base prompt grid: "
    f"{POINTS_PER_SIDE} × {POINTS_PER_SIDE} "
    f"= {len(prompt_grid)} points"
)

# =========================================================
# 6. 单独画 prompt grid
# =========================================================
plt.figure(figsize=(8, 8))

plt.scatter(
    prompt_grid[:, 0],
    prompt_grid[:, 1],
    s=6
)

plt.xlim(0, W)
plt.ylim(H, 0)
plt.gca().set_aspect("equal")

plt.title(
    f"Base Automatic Prompt Grid "
    f"({POINTS_PER_SIDE} x {POINTS_PER_SIDE})"
)

plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "prompt_grid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 7. 画 mask overlap count
# =========================================================
def save_overlap_map(masks, name):
    overlap = masks.astype(np.int32).sum(axis=0)

    plt.figure(figsize=(8, 8))

    im = plt.imshow(overlap)

    plt.colorbar(
        im,
        label="Number of masks covering each pixel"
    )

    plt.title(
        f"{name} Mask Overlap Count"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_overlap_count.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_overlap_map(masks_A, "A")
save_overlap_map(masks_B, "B")

# =========================================================
# 8. 画所有 mask 轮廓
# =========================================================
def save_all_contours(masks, name):
    plt.figure(figsize=(8, 8))

    if measure is not None:
        for i, mask in enumerate(masks):

            contours = measure.find_contours(
                mask.astype(float),
                0.5
            )

            for contour in contours:
                plt.plot(
                    contour[:, 1],
                    contour[:, 0],
                    linewidth=1
                )

    else:
        # 如果没有 skimage，就直接显示 mask 叠加数量
        overlap = masks.astype(np.int32).sum(axis=0)
        plt.imshow(overlap)

    plt.xlim(0, W)
    plt.ylim(H, 0)

    plt.gca().set_aspect("equal")

    plt.title(
        f"All SAM Mask Contours - {name}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_all_mask_contours.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


save_all_contours(masks_A, "A")
save_all_contours(masks_B, "B")

# =========================================================
# 9. 判断 npz 中是否保存了真实 prompt coordinates
# =========================================================
has_prompt_A = "point_coords_A" in data.files
has_prompt_B = "point_coords_B" in data.files

print("\nPrompt metadata:")
print("point_coords_A exists:", has_prompt_A)
print("point_coords_B exists:", has_prompt_B)

if has_prompt_A:
    print(
        "point_coords_A shape:",
        data["point_coords_A"].shape
    )

if has_prompt_B:
    print(
        "point_coords_B shape:",
        data["point_coords_B"].shape
    )

# =========================================================
# 10. 画单个 mask + 基础 prompt grid
# =========================================================
def save_mask_with_grid(
    masks,
    prefix,
    out_dir,
    point_coords=None
):
    for i, mask in enumerate(masks):

        plt.figure(figsize=(8, 8))

        # mask
        plt.imshow(
            mask.astype(np.float32),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # 所有基础 prompt 点
        plt.scatter(
            prompt_grid[:, 0],
            prompt_grid[:, 1],
            s=5,
            alpha=0.35
        )

        # 如果有真实 prompt point
        if point_coords is not None:

            p = np.asarray(point_coords[i])

            # 兼容 [[x,y]] 或 [x,y]
            p = p.reshape(-1, 2)

            plt.scatter(
                p[:, 0],
                p[:, 1],
                s=100,
                marker="*"
            )

        if point_coords is None:
            title = (
                f"{prefix} Mask {i} + Base Prompt Grid\n"
                "(grid points are NOT the exact prompt "
                "responsible for this mask)"
            )
        else:
            title = (
                f"{prefix} Mask {i} + Prompt Point"
            )

        plt.title(title)
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                f"{prefix}_mask_{i:03d}_with_grid.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


point_coords_A = (
    data["point_coords_A"]
    if has_prompt_A else None
)

point_coords_B = (
    data["point_coords_B"]
    if has_prompt_B else None
)

save_mask_with_grid(
    masks_A,
    "A",
    os.path.join(
        OUTPUT_DIR,
        "A_mask_with_grid"
    ),
    point_coords_A
)

save_mask_with_grid(
    masks_B,
    "B",
    os.path.join(
        OUTPUT_DIR,
        "B_mask_with_grid"
    ),
    point_coords_B
)

# =========================================================
# 11. 输出 mask 基本信息
# =========================================================
def print_mask_info(
    masks,
    prefix,
    areas=None,
    bboxes=None,
    ious=None,
    stabilities=None
):
    print(f"\n===== {prefix} MASK INFO =====")

    for i, mask in enumerate(masks):

        actual_area = int(mask.sum())

        print(f"\n{prefix} mask {i}")

        print(
            "actual decoded area:",
            actual_area
        )

        if areas is not None:
            print(
                "saved area:",
                areas[i]
            )

        if bboxes is not None:
            print(
                "bbox:",
                bboxes[i]
            )

        if ious is not None:
            print(
                "SAM predicted IoU:",
                ious[i]
            )

        if stabilities is not None:
            print(
                "stability score:",
                stabilities[i]
            )


print_mask_info(
    masks_A,
    "A",
    data["area_A"]
    if "area_A" in data.files else None,
    data["bbox_A"]
    if "bbox_A" in data.files else None,
    data["iou_A"]
    if "iou_A" in data.files else None,
    data["stability_A"]
    if "stability_A" in data.files else None
)

print_mask_info(
    masks_B,
    "B",
    data["area_B"]
    if "area_B" in data.files else None,
    data["bbox_B"]
    if "bbox_B" in data.files else None,
    data["iou_B"]
    if "iou_B" in data.files else None,
    data["stability_B"]
    if "stability_B" in data.files else None
)

print(
    f"\nDone. Results saved to: {OUTPUT_DIR}"
)