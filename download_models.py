"""
Google Drive에서 모델 파일을 다운로드하는 스크립트.
Render 배포 시 앱 시작 전에 자동 실행됩니다.

지원 방식:
  - GDRIVE_MODEL_FILE_ID: 단일 tar.gz 파일 다운로드
  - GDRIVE_MODEL_FOLDER_ID: 폴더 전체 다운로드
"""
import os
import sys
import tarfile
import tempfile
import shutil

# ---------- 설정 ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LATEST_DIR = os.path.join(MODELS_DIR, "latest")

# Google Drive IDs (환경 변수에서 가져옴)
GDRIVE_FILE_ID = os.getenv("GDRIVE_MODEL_FILE_ID", "")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_MODEL_FOLDER_ID", "")


def _check_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("❌ gdown이 설치되어 있지 않습니다. pip install gdown 을 실행해주세요.")
        sys.exit(1)


def download_file_from_gdrive(file_id: str, output_path: str):
    """gdown을 사용하여 Google Drive에서 단일 파일 다운로드."""
    gdown = _check_gdown()
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"📥 Google Drive에서 모델 파일 다운로드 중... (File ID: {file_id[:12]}...)")
    gdown.download(url, output_path, quiet=False)
    print(f"✅ 다운로드 완료: {output_path}")


def download_folder_from_gdrive(folder_id: str, output_dir: str):
    """gdown을 사용하여 Google Drive 폴더 전체 다운로드."""
    gdown = _check_gdown()
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"📥 Google Drive 폴더에서 모델 다운로드 중... (Folder ID: {folder_id[:12]}...)")
    gdown.download_folder(url, output=output_dir, quiet=False)
    print(f"✅ 폴더 다운로드 완료: {output_dir}")


def extract_models(tar_path: str, target_dir: str):
    """tar.gz 파일을 target_dir에 압축 해제."""
    print(f"📦 모델 압축 해제 중... → {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print("✅ 압축 해제 완료!")


def _has_model_files(directory: str) -> bool:
    """디렉토리에 모델 파일(phase 폴더)이 있는지 확인."""
    if not os.path.isdir(directory):
        return False
    entries = os.listdir(directory)
    phase_dirs = [d for d in entries if d.startswith("phase")]
    return len(phase_dirs) > 0


def main():
    # 이미 모델이 있으면 스킵
    if _has_model_files(LATEST_DIR):
        phase_dirs = [d for d in os.listdir(LATEST_DIR) if d.startswith("phase")]
        print(f"✅ 모델이 이미 존재합니다 ({len(phase_dirs)}개 phase). 다운로드 스킵.")
        return

    # 방식 1: 폴더 ID로 다운로드
    if GDRIVE_FOLDER_ID:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            download_folder_from_gdrive(GDRIVE_FOLDER_ID, tmpdir)

            # 다운로드된 폴더 구조 확인 & latest/ 에 배치
            downloaded = os.listdir(tmpdir)
            print(f"📂 다운로드된 항목: {downloaded}")

            # tar.gz 파일이 있으면 압축 해제
            tar_files = [f for f in downloaded if f.endswith(".tar.gz")]
            if tar_files:
                tar_path = os.path.join(tmpdir, tar_files[0])
                extract_models(tar_path, LATEST_DIR)
            else:
                # 폴더 자체가 모델 파일인 경우 latest/로 복사
                os.makedirs(LATEST_DIR, exist_ok=True)
                for item in downloaded:
                    src = os.path.join(tmpdir, item)
                    dst = os.path.join(LATEST_DIR, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

        # 결과 확인
        if _has_model_files(LATEST_DIR):
            print("🎉 모델 준비 완료! 앱을 시작할 수 있습니다.")
        else:
            print("⚠️  모델 파일을 찾을 수 없습니다. 폴더 구조를 확인해 주세요.")
        return

    # 방식 2: 단일 파일 ID로 다운로드
    if GDRIVE_FILE_ID:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "models_latest.tar.gz")
            download_file_from_gdrive(GDRIVE_FILE_ID, tar_path)
            extract_models(tar_path, LATEST_DIR)
        print("🎉 모델 준비 완료! 앱을 시작할 수 있습니다.")
        return

    # 둘 다 없으면
    print("⚠️  GDRIVE_MODEL_FILE_ID 또는 GDRIVE_MODEL_FOLDER_ID 환경 변수가 설정되지 않았습니다.")
    print("   모델 없이 앱을 시작합니다. 일부 기능이 제한될 수 있습니다.")
    os.makedirs(LATEST_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
