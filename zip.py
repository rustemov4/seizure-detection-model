import os
import zipfile

def zip_without_venv(source_dir, archive_path):
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            if 'venv' in dirs:
                dirs.remove('venv')

            for filename in files:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, start=source_dir)
                zf.write(full_path, arcname=relative_path)

if __name__ == "__main__":
    source_dir = "."
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    archive_name = "my_project.zip"

    archive_path = os.path.join(desktop_path, archive_name)

    zip_without_venv(source_dir, archive_path)
    print(f"Created: {archive_path}, excluding 'venv' folder.")
