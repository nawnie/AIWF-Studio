import pathlib, shutil
src = pathlib.Path(r"C:\Users\Shawn\Desktop\AIWF-Studio\aiwf\infrastructure\wan\pipeline.py")
dst = pathlib.Path(r"C:\Users\Shawn\Desktop\AIWF-Studio\pipeline_readable.py")
shutil.copy2(src, dst)
print(f"Copied {src.stat().st_size} bytes to {dst}")
