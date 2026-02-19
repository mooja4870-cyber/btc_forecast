"""
Google Drive에서 모델 파일을 다운로드하는 스크립트.
Render 배포 시 앱 시작 전에 자동 실행됩니다.
"""
import os
import sys
import tarfile
import tempfile

# ---------- 설정 ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LATEST_DIR = os.path.join(MODELS_DIR, "latest")

# Google Drive File ID (환경 변수에서 가져옴)
GDRIVE_FILE_ID = os.getenv("GDRIVE_MODEL_FILE_ID", "")


def download_from_gdrive(file_id: str, output_path: str):
    """gdown을 사용하여 Google Drive에서 파일 다운로드."""
    try:
        import gdown
    except ImportError:
        print("❌ gdown이 설치되어 있지 않습니다. pip install gdown 을 실행해주세요.")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"📥 Google Drive에서 모델 다운로드 중... (File ID: {file_id[:8]}...)")
    gdown.download(url, output_path, quiet=False)
    print(f"✅ 다운로드 완료: {output_path}")


def extract_models(tar_path: str, target_dir: str):
    """tar.gz 파일을 target_dir에 압축 해제."""
    print(f"📦 모델 압축 해제 중... → {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"✅ 압축 해제 완료!")


def main():
    # 이미 모델이 있으면 스킵
    if os.path.isdir(LATEST_DIR) and os.listdir(LATEST_DIR):
        phase_dirs = [d for d in os.listdir(LATEST_DIR) if d.startswith("phase")]
        if phase_dirs:
            print(f"✅ 모델이 이미 존재합니다 ({len(phase_dirs)}개 phase). 다운로드 스킵.")
            return

    # File ID 확인
    if not GDRIVE_FILE_ID:
        print("⚠️  GDRIVE_MODEL_FILE_ID 환경 변수가 설정되지 않았습니다.")
        print("   모델 없이 앱을 시작합니다. 일부 기능이 제한될 수 있습니다.")
        os.makedirs(LATEST_DIR, exist_ok=True)
        return

    # 다운로드 & 압축 해제
    os.makedirs(MODELS_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "models_latest.tar.gz")
        download_from_gdrive(GDRIVE_FILE_ID, tar_path)
        extract_models(tar_path, MODELS_DIR)

    print("🎉 모델 준비 완료! 앱을 시작할 수 있습니다.")


if __name__ == "__main__":
    main()
