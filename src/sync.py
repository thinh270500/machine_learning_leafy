import os
import shutil
import gdown


def download_from_drive(folder_id, local_path="Dataset/Raw"):
    """
    Tải toàn bộ nội dung thư mục Google Drive (folder_id)
    về local theo đúng cấu trúc thư mục con.
    """
    os.makedirs(local_path, exist_ok=True)
    print(f"🔽 Đang tải dữ liệu từ Drive folder ID: {folder_id} ...")

    try:
        gdown.download_folder(
            id=folder_id,
            output=local_path,
            quiet=False,
            use_cookies=False
        )
        print(f"✅ Tải hoàn tất. Dữ liệu lưu tại: {local_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tải từ Drive: {e}")


def rename_images_in_folder(root_dir):
    """
    Duyệt tất cả thư mục con (Tomato, Carrot, ...) và
    đổi tên ảnh theo mẫu <ten_thu_muc><so_thu_tu>.<duoi_anh>.
    """
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()

        for idx, img_name in enumerate(images, start=1):
            ext = os.path.splitext(img_name)[1]
            new_name = f"{class_name.lower()}{idx}{ext}"
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(class_dir, new_name)

            if src != dst and not os.path.exists(dst):
                shutil.move(src, dst)

        print(f"✅ Đã đổi tên {len(images)} ảnh trong thư mục: {class_name}")


if __name__ == "__main__":
    # 🔧 Cấu hình tại đây
    DRIVE_FOLDER_ID = "https://drive.google.com/drive/folders/1tDYQhEZy_WovYko2swNTZcbG8XAC68FQ?usp=sharing"  # ← dán ID Drive folder Raw
    LOCAL_PATH = "Dataset/Raw"

    # 1️⃣ Tải dữ liệu từ Drive
    download_from_drive(DRIVE_FOLDER_ID, LOCAL_PATH)

    # 2️⃣ Đổi tên file ảnh sau khi tải
    rename_images_in_folder(LOCAL_PATH)
